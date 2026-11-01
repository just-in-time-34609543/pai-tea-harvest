"""Build the course report PDF (Japanese) from the experiment results, with figures.

    python scripts/make_report_pdf.py   -> docs/report/report.pdf, figures in docs/report/figures/

Optional: REPORT_GITHUB_URL and REPORT_YOUTUBE_URL add the submission links to the first page.
REPORT_FONT (or TEA_FONT, or UE_ROOT) sets the Japanese font file.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT = os.environ.get("REPORT_FONT") or os.environ.get("TEA_FONT") or os.path.join(os.environ.get("UE_ROOT", "UE_5.7.4"), "Engine/Content/Slate/Fonts/DroidSansFallback.ttf")
FIG = ROOT / "docs" / "report" / "figures"
EV = ROOT / "output" / "adeept" / "eval"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
pdfmetrics.registerFont(TTFont("JP", FONT))


def rates(name):
    d = json.loads((EV / name / "results.json").read_text())["success"]
    return [d[k] * 100 for k in ("harvest", "harvest_left", "harvest_middle", "harvest_right")], d["mean"] * 100


def fig_system():
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off")
    boxes = [("アーチ上の\nRGB-Dカメラ4台\n＋手首カメラ", "#dfe9d6"), ("点群の統合\n（台座標、2048点\nxyz＋色）", "#e6eef6"),
             ("AI：点群＋指示文\n→ 目標の新芽の\n3D把持点", "#f6e6d6"), ("逆運動学で\n接近→把持→引抜\n→かごへ→解放", "#ece2f2"), ("成功判定\n・記録", "#eeeeee")]
    for i, (t, c) in enumerate(boxes):
        x = 0.02 + i * 0.198
        ax.add_patch(FancyBboxPatch((x, 0.2), 0.16, 0.6, boxstyle="round,pad=0.01", fc=c, ec="#555"))
        ax.text(x + 0.08, 0.5, t, ha="center", va="center", fontsize=11)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.198, 0.5), xytext=(x + 0.165, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.05, "物理：MuJoCo 3.17　／　描画：Unreal Engine 5.7（姿勢を毎ステップ送って描画）", ha="center", fontsize=10, color="#444")
    fig.savefig(FIG / "fig_system.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_results():
    names = [("ue_mj_target", "新芽位置推定＋IK\n（Unreal＋MuJoCo学習）"), ("ue_target_from_mj", "新芽位置推定＋IK\n（MuJoCo学習→Unreal）"),
             ("ue_target", "新芽位置推定＋IK\n（Unrealのみ学習）"), ("ue_pcvla_v4", "点群VLA\n（Unrealのみ学習）"), ("ue_mj_pcvla", "点群VLA\n（Unreal＋MuJoCo学習）")]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    labels = ["任意", "左", "真ん中", "右"]
    w = 0.16
    for k, (n, lab) in enumerate(names):
        r, m = rates(n)
        ax.bar([i + (k - 2) * w for i in range(4)], r, w, label=f"{lab.replace(chr(10), ' ')}  平均{m:.1f}%")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 105); ax.set_ylabel("成功率（%）")
    ax.set_title("Unreal描画での成功率（指示ごと、保留シード各10試行）"); ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=.3)
    fig.savefig(FIG / "fig_results.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def jpg(src, name, width=1400):
    # 元の画像が無い環境では、既に docs/report/figures にある図をそのまま使う
    if not Path(src).exists() and (FIG / name).exists():
        return
    im = Image.open(src).convert("RGB"); im.thumbnail((width, width)); im.save(FIG / name, quality=85)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_system(); fig_results()
    jpg(ROOT / "output/adeept/report_src/video_frame.png", "fig_video_frame.jpg")
    jpg(ROOT / "output/adeept/report_src/arch_cams_with_cloud.png", "fig_cameras.jpg")
    jpg(ROOT / "output/adeept/ue_preview/v5/tea_inside_row.png", "fig_greenhouse.jpg")
    st = lambda size=10.5, lead=None, bold=False: ParagraphStyle("s", fontName="JP", fontSize=size, leading=lead or size * 1.6)
    H1, H2, P = st(17), st(13), st(10.2)
    S = []
    add = lambda t, s=P: S.append(Paragraph(t, s))
    img = lambda f, w: S.append(RLImage(str(FIG / f), width=w, height=w * Image.open(FIG / f).size[1] / Image.open(FIG / f).size[0]))
    add("点群AIによる茶の新芽の選択摘採ロボット", H1)
    add("Physical AI応用1 最終課題　説明資料（数値はすべてシミュレーションの結果で、実機・実茶樹での結果ではない）", st(9.5))
    links = [(k, os.environ.get(v)) for k, v in (("動画", "REPORT_YOUTUBE_URL"), ("コード", "REPORT_GITHUB_URL")) if os.environ.get(v)]
    if links:
        add("　".join(f"{k}：{u}" for k, u in links), st(9.5))
    S.append(Spacer(1, 4 * mm))
    add("1. 課題の概要と目的", H2)
    add("茶の選択摘採では、摘む若い新芽と、残す成葉・芽・枝を見分け、枝を傷付けずに1本ずつ摘み取る必要がある。本課題では、ビニールハウスの天井レールを走る台車から吊った台に小型ロボットアーム（Adeept ADA031）を載せ、ハウスの骨組み（アーチ）に付けたカメラの点群から、言葉で指定した新芽をAIが特定し、摘んで回収箱へ入れるまでを自動で行うシステムを、物理シミュレーション（MuJoCo）と写実的なレンダリング（Unreal Engine）で構築・評価した。")
    img("fig_greenhouse.jpg", 165 * mm)
    add("図1　実寸のパイプハウスと茶樹（Unreal Engine 5.7、Lumen）。茶樹・ハウス・地面は写真を参考にプログラムで生成した。", st(9))
    add("2. タスク設定と成功条件", H2)
    add("ハウスは実運用想定の幅5.7m×長さ10m×高さ3.1m（アーチ12本）、茶樹は高さ1.5〜2.1m。台のまわりの樹冠から新芽が3本出ており、指示文は「新芽を1本摘んで、かごに入れて」「左の…」「真ん中の…」「右の…」の4種類。指定した新芽（任意の場合はどれか1本だけ）を摘み、かごの中へ入れ、他の新芽は株に残し、グリッパーを開いていれば成功。新芽は株に拘束されており、指で握って上へ引いたときだけ外れる（切断・損傷は未モデル化）。")
    add("3. システム構成と処理の流れ", H2)
    img("fig_system.png", 170 * mm)
    add("図2　処理の流れ。MuJoCoで計算した全物体・カメラの姿勢をUnrealへ毎ステップ送り、RGBと深度を描画する。", st(9))
    img("fig_cameras.jpg", 165 * mm)
    add("図3　アーチ上の4台のカメラ（上段3枚と下段左）、手首カメラ（下段中央）、統合した点群を上から見た図とAIの推定把持点（下段右、赤丸）。", st(9))
    S.append(PageBreak())
    add("4. AIを活用した箇所と、その必要性", H2)
    add("新芽の位置は株ごと・試行ごとに変わり、茶樹の葉や枝と同じ緑色で重なって見える。決まった座標を巡回するだけでは指定の新芽を摘めないため、カメラの点群と指示文から「どの新芽を、どこで掴むか」をAIが判断する。")
    tbl = [["方式", "入力", "出力", "役割"],
           ["新芽位置推定＋IK", "点群、指示文", "指定新芽の3D把持点", "AIが摘採候補を特定し座標を求める。動作は逆運動学"],
           ["点群VLA", "点群、指示文", "16ステップ分の関節目標", "AIが動作まで直接出す（flow matching）"],
           ["SmolVLA（比較）", "画像2枚、指示文、関節", "関節目標のチャンク", "事前学習済みVLAの微調整"]]
    cell = st(8.8)
    tbl = [[Paragraph(c, cell) for c in row] for row in tbl]
    t = Table(tbl, colWidths=[32 * mm, 32 * mm, 38 * mm, 68 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "JP", 8.8), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8efe4"))]))
    S.append(t); S.append(Spacer(1, 3 * mm))
    add("点群エンコーダは最遠点サンプリングで128中心・各32近傍の局所PointNet＋中心座標のFourier埋め込み。指示文はSmolVLAの言語モデル（SmolVLM2）を凍結して特徴化。学習データはAdeeptの逆運動学によるお手本の成功エピソードのみ（Unreal 200話、MuJoCo 400話）。")
    add("5. 結果", H2)
    img("fig_results.png", 170 * mm)
    add("図4　Unreal描画での成功率。新芽位置推定＋IK（Unreal＋MuJoCo学習）が平均97.5%（推定誤差の中央値1.5〜2.5mm）、点群VLAは75%。", st(9))
    add("MuJoCo描画（茶樹の簡素版）では、新芽位置推定＋IKが97.5%、点群VLA（関節状態なし）75%、点群VLA（関節状態あり）15%、SmolVLA 5%。簡素なMuJoCo画面で学習した位置推定AIは、写実的なUnreal画面でもそのまま95%で動いた。点群VLAは、MuJoCoのデータを加えても伸びなかった（75%→70%。各10試行の差で、誤差の範囲）。")
    img("fig_video_frame.jpg", 160 * mm)
    add("図5　動画の一場面（左の新芽を摘んでかごへ入れたところ）。全景、アーチ上のカメラ、手首カメラ、点群とAIの推定把持点（赤丸）。", st(9))
    add("6. 工夫した点と、前の版から発展させた点", H2)
    for s in ["前の版（同じリポジトリの main ブランチ、v7）は、天井吊りの2関節アームと球殻グリッパーで、決まった位置の葉を順に回収するもので、AIは使っていなかった。本版では、言葉で指定した新芽をAIがカメラの点群から探して摘む形に発展させた。",
              "物理はMuJoCo、描画はUnreal Engineで分担し、MuJoCoで計算した全物体とカメラの姿勢を毎ステップUnrealへ送って、画面のないGPUサーバーで写実的なRGB画像と深度を得る仕組みを作った。UnrealRoboticsLab（MuJoCoとUnrealの連携）の考え方を参考にしたが、そのままでは画面なし環境でカメラ画像が出ず、複数起動で停止したため、姿勢を送る方式に置き換えた。",
              "茶樹・パイプハウス・地面は、実際の茶樹の写真を参考にプログラムで生成した。外部の3Dアセットは使っていない。",
              "AIで「どの新芽か・どこか」を推定し、動作は運動学で計画する役割分担型と、動作までAIが出すエンドツーエンド型（点群VLA、SmolVLA）を、同じ条件で比べた。",
              "簡素なMuJoCo描画で学習したAIを、写実的なUnreal描画にそのまま使えるか確かめ（95%）、両方のデータを混ぜて97.5%にした。"]:
        add("・" + s)
    add("7. 実装と使用した主な技術", H2)
    add("物理シミュレーション MuJoCo 3.17.0、描画 Unreal Engine 5.7.4（Lumen、画面なしのVulkan）、学習 PyTorch 2.10.0、比較用VLA LeRobot 0.4.4 のSmolVLA、指示文の特徴 SmolVLM2-500M-Video-Instruct。アームはAdeept ADA031（5自由度、2本指グリッパー）のシミュレーションモデル。動画は10fpsのシミュレーションを10fpsで書き出した等速で、倍速部分はない（タイトルと終わりの画面は静止画）。")
    add("8. 失敗例と考察", H2)
    for s in ["デモの記録を成功判定の瞬間で打ち切ると、グリッパーを開き切る動作が学習データに入らず、方策が「離す直前」で止まった。記録をお手本の最後まで延ばして解決した。",
              "関節状態を入力に含めた点群VLAは、状態から次の動作を当てる近道を覚え、新芽の位置を読まずに空振りした（15%）。状態入力を外すと点群から位置を読むようになり75%に改善した。",
              "少数のデモでは、動作全体を丸ごと学ぶより、「どの新芽か・どこか」をAIが推定し、動作は既知の運動学で計画する方が確実（97.5%）。実運用でも、認識と動作計画を分けた方が検証・保守しやすい。",
              "Unrealのみのデータ（200話）で学習した位置推定は82.5%で、MuJoCoのデータを加えると97.5%に向上した。点群は形と色が主で、画面の質感の差に強い。"]:
        add("・" + s)
    add("9. 制約と今後", H2)
    add("新芽は剛体の模型で、切断・柔らかさ・損傷は扱わない。新芽は株の上部にあり遮蔽は少ない。実写茶葉の認識は未実施で、摘む／残すの判断は指示文の位置指定で代用している。実機への移行には、カメラ校正と実測点群での追加学習が必要。")
    add("10. 参考にしたもの・使用した資産", H2)
    for s in ["MuJoCo（Google DeepMind、Apache-2.0）、Unreal Engine 5.7（Epic Games、Unreal Engine EULA）",
              "UnrealRoboticsLab：MuJoCoとUnreal Engineを連携させる考え方を参考にした（本リポジトリにはコードを含めていない）",
              "LeRobot と SmolVLA、SmolVLM2（Hugging Face、Apache-2.0）。重みはリポジトリに含めていない",
              "Adeept ADA031 のシミュレーションモデルと逆運動学：本プロジェクト用に作成し、adeept_sim/ に同梱（キットの説明書の寸法から作成）",
              "茶樹の参照写真：形と色を合わせるための参考に使った。学習データには使っておらず、リポジトリにも含めない"]:
        add("・" + s)
    doc = SimpleDocTemplate(str(ROOT / "docs" / "report" / "report.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    doc.build(S)
    print("pdf", ROOT / "docs" / "report" / "report.pdf")


if __name__ == "__main__":
    main()
