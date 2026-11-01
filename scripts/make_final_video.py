"""Final course video (Unreal render): AI picks the instructed tea shoot from the fused point cloud, the arm harvests it.

    python scripts/make_final_video.py --ckpt output/adeept/pcvla/ue_target/ckpt.pt --port 5600 --out output/adeept/final_video.mp4

Needs a render server started with TEA_CAMS="cine,camera1,camera2,camera3,camera4,camera5" (cine = 1280x720 overview).
Layout (1280x720): orbiting overview | arch camera | wrist camera | point cloud with the AI's predicted grasp point,
captions in Japanese. Real time (10 fps simulation, 10 fps video); the title and ending cards are stills.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pcvla"))
import harvest_env as H  # noqa: E402
import pointcloud as PC  # noqa: E402
import train as T  # noqa: E402
from make_tea_demos import platform_origin  # noqa: E402
from target_net import TargetNet, CROP  # noqa: E402
from ue_render_client import UERenderClient  # noqa: E402

FONT = os.environ.get("TEA_FONT") or os.environ.get("REPORT_FONT") or os.path.join(os.environ.get("UE_ROOT", "UE_5.7.4"), "Engine/Content/Slate/Fonts/DroidSansFallback.ttf")
FONT_INDEX = int(os.environ.get("TEA_FONT_INDEX", os.environ.get("REPORT_FONT_INDEX", "0")))  # 日本語フォント（Unreal Engine に入っているもの）
JA = {"harvest": "茶の成葉を1枚摘んで、かごに入れて", "harvest_left": "左の成葉を摘んで、かごに入れて",
      "harvest_middle": "真ん中の成葉を摘んで、かごに入れて", "harvest_right": "右の成葉を摘んで、かごに入れて"}
PHASES = ["成葉の上へ接近", "成葉を挟むように下降", "グリッパーを閉じて保持", "引き上げて成葉を外す（葉柄は株に残る）", "かごの上へ移動", "グリッパーを開いて回収"]


def font(sz):
    return ImageFont.truetype(FONT, sz, index=FONT_INDEX)


def card(lines, sizes, w=1920, h=1080, bg=(16, 28, 16)):
    im = Image.new("RGB", (w, h), bg); d = ImageDraw.Draw(im)
    y = h // 2 - sum(s + 18 for s in sizes) // 2
    for text, sz in zip(lines, sizes):
        f = font(sz); tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2, y), text, font=f, fill=(235, 240, 225)); y += sz + 18
    return np.array(im)


def cloud_panel(cloud, pred, truth, size=(640, 360)):
    """Top view of the fused cloud for video (survives YouTube compression): thick discs, black background, zoomed to
    the leaf/basket region, the AI grasp point as a thick yellow ring with a cross."""
    w, h = size; top = 40
    im = Image.new("RGB", size, (0, 0, 0)); d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = -0.03, 0.22, -0.16, 0.22                      # platform frame (m)
    s = min((h - top - 8) / (x1 - x0), (w - 12) / (y1 - y0))
    ox = (w - (y1 - y0) * s) / 2
    xy = lambda p: (ox + (p[1] - y0) * s, top + (x1 - p[0]) * s)
    for p in cloud[np.argsort(cloud[:, 2])]:
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            continue
        x, y = xy(p); c = tuple(int(min(255, 60 + 255 * 1.4 * v)) for v in p[3:6])
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
    x, y = xy(pred)
    d.ellipse([x - 22, y - 22, x + 22, y + 22], outline=(255, 230, 0), width=6)
    d.line([x - 32, y, x + 32, y], fill=(255, 230, 0), width=4); d.line([x, y - 32, x, y + 32], fill=(255, 230, 0), width=4)
    d.text((10, 6), "点群（上から）と AIの推定把持点", font=font(24), fill=(255, 255, 255))
    return np.array(im)


ORBIT_DEG_PER_S = float(os.environ.get("TEA_ORBIT_DEG_PER_S", "0"))


def cine_pose(origin, t, focus=None):
    """Overview camera: fixed (default) or a slow orbit (TEA_ORBIT_DEG_PER_S), front-left, close enough to see the
    target leaf and the basket (focus = platform-frame point between them)."""
    a = math.radians(-50 - 35 + ORBIT_DEG_PER_S * t) if ORBIT_DEG_PER_S else math.radians(-50)
    tgt = origin + (np.array([0.10, 0.04, 0.03]) if focus is None else np.asarray(focus))
    eye = tgt + np.array([0.42 * math.cos(a), 0.42 * math.sin(a), 0.22])
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return [*eye, *f, *u, 55.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/adeept/pcvla/ue_target/ckpt.pt")
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--out", default="output/adeept/final_video.mp4")
    p.add_argument("--seeds", default="100003,100001,100002,100004")
    a = p.parse_args()
    T.CROP = np.array([float(v) for v in CROP.split(",")]).reshape(3, 2)
    lang = torch.load(Path(__file__).resolve().parents[1] / "output" / "adeept" / "pcvla" / "lang.pt")
    net = TargetNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    env = H.HarvestEnv(renderer="none", size=(240, 320))
    frames = [card(["点群AIによる茶の成葉の選択摘採ロボット", "Physical AI応用1 最終課題（シミュレーション）",
                    "ハウスの骨に付けた4台のRGB-Dカメラ＋手首カメラ → 点群 → 指示された成葉をAIが特定 → アームで摘んで回収"],
                   [66, 38, 30])] * 30
    results = []
    for task, seed in zip(H.TASKS, map(int, a.seeds.split(","))):
        origin = platform_origin(np.random.default_rng(seed)); env.origin = np.asarray(origin)
        ue = UERenderClient(origin=origin, port=a.port, depth=True); env.ue = ue
        env.reset(seed, task=task)
        r = np.random.default_rng(seed)
        cloud = ue.cloud(env, r)
        with torch.no_grad():
            pred = net(torch.from_numpy(T.crop_resample(cloud, r))[None].float().cuda(), lang[env.language][None].float().cuda())[0].cpu().numpy()
        chosen = int(np.argmin([np.linalg.norm(pred - s) for s in env.spots]))
        if H.TASKS[task].target == "any":
            env.target = chosen
        err = float(np.linalg.norm(pred - env.spots[env.target]))
        env.spots = list(env.spots); env.spots[env.target] = pred.astype(float)
        ex = H.Expert(env); done, success, k = False, False, 0
        cp = cloud_panel(ue.cloud(env, r, n=16000), pred, env.spots[env.target])
        while not done:
            bodies, cams = ue.poses(env)
            for c in cams.values():
                c[9] = float(np.degrees(2 * np.arctan(np.tan(np.radians(c[9]) / 2) * 320 / 240)))
            cams["cine"] = cine_pose(np.asarray(origin), k / 10, focus=0.5 * (np.asarray(env.spots[env.target]) + np.array([*env.box_xy, 0.03])))
            files = ue._request({"bodies": bodies, "cams": {n: cams[n] for n in ("cine", "camera1", "camera2")}, "out": f"{ue.dir}/v{k:05d}"})["files"]
            im = {n: imageio.imread(f)[..., :3] for n, f in files.items()}
            for f in files.values():
                os.remove(f)
            canvas = Image.new("RGB", (1920, 1080), (12, 18, 12))
            canvas.paste(Image.fromarray(im["cine"]).resize((1280, 720)), (0, 0))
            canvas.paste(Image.fromarray(im["camera1"]).resize((640, 360)), (1280, 0))
            canvas.paste(Image.fromarray(im["camera2"]).resize((640, 360)), (1280, 360))
            canvas.paste(Image.fromarray(cp), (1280, 720))
            d = ImageDraw.Draw(canvas)
            for (tx, ty), label in (((1290, 6), "アーチ上のカメラ"), ((1290, 366), "手首カメラ")):
                d.rectangle([tx - 6, ty - 2, tx + 230, ty + 34], fill=(0, 0, 0)); d.text((tx, ty), label, font=font(26), fill=(255, 255, 255))
            d.text((24, 740), f"指示：「{JA[task]}」", font=font(44), fill=(245, 245, 225))
            d.text((24, 806), f"AI：点群と指示文から目標の成葉を特定（推定誤差 {err * 1000:.1f} mm）", font=font(32), fill=(200, 235, 200))
            phase = PHASES[min(ex.index, len(PHASES) - 1)] if ex.index < len(ex.plan) else "回収完了"
            d.text((24, 862), f"動作：{phase}", font=font(32), fill=(230, 230, 205))
            d.text((24, 1030), f"t = {env.ticks / 10:4.1f} s（実時間）   MuJoCo物理 + Unreal Engine 5.7描画", font=font(22), fill=(165, 175, 165))
            frames.append(np.array(canvas))
            _, success, done = env.step(ex.act()); k += 1
        ue.close()
        results.append((task, success, err))
        res = "成功：成葉をかごに回収、葉柄と他の成葉は株に残っています" if success else "失敗"
        frames += [np.array(Image.fromarray(frames[-1]).point(lambda v: v * 0.55))] * 1
        last = Image.fromarray(frames[-1]); ImageDraw.Draw(last).text((24, 930), res, font=font(40), fill=(255, 230, 120))
        frames += [np.array(last)] * 15
    summary = [f"{JA[t]}：{'成功' if s else '失敗'}（推定誤差 {e * 1000:.1f} mm）" for t, s, e in results]
    frames += [card(["結果（この動画の4試行）"] + summary + ["評価（保留シード各10試行）の成功率はレポートを参照"], [54] + [34] * len(summary) + [28])] * 40
    imageio.mimsave(a.out, frames, fps=10, quality=8)
    print("video", a.out, len(frames) / 10, "s", results)


if __name__ == "__main__":
    main()
