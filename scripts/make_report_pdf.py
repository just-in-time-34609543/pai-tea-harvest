"""Build the course report (Japanese) for the mature-leaf version: docs/report/report.pdf and docs/report/report.md,
with figures in docs/report/figures/. All numbers are read from the result files in output/adeept/eval/.

    python scripts/make_report_pdf.py

Optional: REPORT_GITHUB_URL and REPORT_YOUTUBE_URL add the submission links to the first page.
REPORT_FONT (or TEA_FONT, or UE_ROOT) sets the Japanese font file; REPORT_FONT_INDEX picks the face of a .ttc.
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
FONT_INDEX = int(os.environ.get("REPORT_FONT_INDEX", "0"))
FIG = ROOT / "docs" / "report" / "figures"
EV = ROOT / "output" / "adeept" / "eval"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
pdfmetrics.registerFont(TTFont("JP", FONT, subfontIndex=FONT_INDEX))
TASKS = ("harvest", "harvest_left", "harvest_middle", "harvest_right")


def res(name):
    d = json.loads((EV / name / "results.json").read_text())
    s = [d["success"][k] * 100 for k in TASKS]
    e = [d["median_err_m"][k] * 1000 for k in TASKS] if "median_err_m" in d else None
    return s, d["success"]["mean"] * 100, e


def para():
    d = json.loads((EV / "paraphrase_final_mj" / "summary.json").read_text())
    return d["overall"]["original"] * 100, d["overall"]["paraphrase"] * 100, d["overall"]["relative_drop"] * 100


def fig_system():
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off")
    boxes = [("アーチ上の\nRGB-Dカメラ4台\n＋手首カメラ", "#dfe9d6"), ("点群の統合\n（台座標、2048点\nxyz＋色）", "#e6eef6"),
             ("AI：点群＋指示文\n→ 目標の成葉の\n3D把持点", "#f6e6d6"), ("逆運動学で\n接近→把持→引抜\n→かごへ→解放", "#ece2f2"), ("成功判定\n・記録", "#eeeeee")]
    for i, (t, c) in enumerate(boxes):
        x = 0.02 + i * 0.198
        ax.add_patch(FancyBboxPatch((x, 0.2), 0.16, 0.6, boxstyle="round,pad=0.01", fc=c, ec="#555"))
        ax.text(x + 0.08, 0.5, t, ha="center", va="center", fontsize=11)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.198, 0.5), xytext=(x + 0.165, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.05, "物理：MuJoCo 3.17　／　描画：Unreal Engine 5.7（姿勢を毎ステップ送って描画）", ha="center", fontsize=10, color="#444")
    fig.savefig(FIG / "fig_system.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_results(mj, ue, po, pp):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    labels = ["任意", "左", "真ん中", "右"]
    w = 0.38
    ax.bar([i - w / 2 for i in range(4)], ue[0], w, label=f"Unreal描画（写実的な茶樹） 平均{ue[1]:.1f}%", color="#3c7a3c")
    ax.bar([i + w / 2 for i in range(4)], mj[0], w, label=f"MuJoCo描画（簡素） 平均{mj[1]:.1f}%", color="#9cc49c")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 105); ax.set_ylabel("成功率（%）")
    ax.set_title("成葉の位置推定＋IKの成功率（指示ごと、学習に使わない乱数シードで各10試行）")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=.3)
    fig.savefig(FIG / "fig_results.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def jpg(src, name, width=1400):
    if not Path(src).exists() and (FIG / name).exists():
        return
    im = Image.open(src).convert("RGB"); im.thumbnail((width, width)); im.save(FIG / name, quality=85)


def pct(v):
    return f"{v:.0f}%" if abs(v - round(v)) < 1e-6 else f"{v:.1f}%"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    mj, ue = res("leaf_mj_target"), res("leaf_ue_mj_target")
    po, pp, pdrop = para()
    fig_system(); fig_results(mj, ue, po, pp)
    jpg(ROOT / "output/adeept/report_src/leaf_video_frame.png", "fig_video_frame.jpg")
    jpg(ROOT / "output/adeept/report_src/leaf_arch_cams_with_cloud.png", "fig_cameras.jpg")
    jpg(ROOT / "output/adeept/ue_preview/v5/tea_inside_row.png", "fig_greenhouse.jpg")
    links = [(k, os.environ.get(v)) for k, v in (("動画", "REPORT_YOUTUBE_URL"), ("コード", "REPORT_GITHUB_URL")) if os.environ.get(v)]
    err_rng = f"{min(ue[2]):.1f}〜{max(ue[2]):.1f}mm"
    row = lambda name, r: [name] + [pct(v) for v in r[0]] + [pct(r[1])]

    T = {}   # text blocks shared by the PDF and the Markdown
    T["title"] = "点群AIによる茶の成葉の選択摘採ロボット"
    T["sub"] = "Physical AI応用1 最終課題　説明資料（数値はすべてシミュレーションの結果で、実機・実茶樹での結果ではない）"
    T["s1"] = ("茶の成葉を収穫するには、摘む葉と残す枝・葉柄を見分け、枝を傷付けずに1枚ずつ摘み取る必要がある。本課題では、ビニールハウスの天井レールを走る台車から吊った台に小型ロボットアーム（Adeept ADA031）を載せ、"
               "ハウスの骨組み（アーチ）に付けたカメラの点群から、言葉で指定した成葉をAIが特定し、1枚ずつ摘んで回収箱へ入れるまでを自動で行うシステムを、物理シミュレーション（MuJoCo）と写実的なレンダリング（Unreal Engine）で構築・評価した。")
    T["fig1"] = "図1　実寸のパイプハウスと茶樹（Unreal Engine 5.7、Lumen）。茶樹・ハウス・地面は写真を参考にプログラムで生成した。"
    T["s2"] = ["場面：実運用想定のパイプハウス（幅5.7m×長さ10m×高さ3.1m、アーチ12本）に高さ1.5〜2.1mの茶樹。天井レールの台車から吊った台に、Adeept ADA031（5自由度、2本指グリッパー）を載せる。",
               "1回の試行：台のまわりの樹冠から、摘む対象の成葉（緑の葉）が3枚立ち上がっている。指示文は4種類で固定（英語。「成葉を1枚摘んでかごへ」「左の成葉を…」「真ん中の成葉を…」「右の成葉を…」）。",
               "成功：指定した成葉（任意の場合はどれか1枚だけ）を摘み、かごの中へ入れ、他の成葉は株に残り、グリッパーを開いている。葉柄と枝は株に残る。",
               "摘み取りの模擬：成葉は株（葉柄）に拘束されており、指で挟んで上へ引いたときだけ外れる。摘めたかどうかは、指の接触・グリッパーの角度・引き上げで判定し、どの葉でも同じ条件（切断・柔らかさ・損傷は未モデル化）。"]
    T["fig2"] = "図2　処理の流れ。MuJoCoで計算した全物体・カメラの姿勢をUnrealへ毎ステップ送り、RGBと深度を描画する。"
    T["fig3"] = "図3　アーチ上のカメラ（左、4台のうち1台）、手首カメラ（中）、4台＋手首の深度から統合した点群を上から見た図とAIの推定把持点（右、黄色の丸）。"
    T["s4"] = ("成葉の位置は株ごと・試行ごとに変わり、茶樹のほかの葉や枝と同じ緑色で重なって見える。決まった座標を巡回するだけでは指定の成葉を摘めないため、カメラの点群と指示文から「どの成葉を、どこで掴むか」をAIが推定する。"
               "動作の計画（接近・把持・引き抜き・かごへの移動・解放）は、AIの推定点だけを使って逆運動学で行う。かごへの落とし位置の補正も、指を閉じた時点の推定点とグリッパー先端の差から求め、シミュレーターの正解の位置は使わない。")
    T["s4b"] = ("AI：点群（最遠点サンプリングで128中心、各32近傍の局所PointNet＋中心座標のFourier埋め込み）と指示文の特徴（SmolVLM2-500M の言語モデルを凍結、文全体の平均）を Transformer で統合し、目標の成葉の3D位置を回帰する。"
                "学習データは、逆運動学によるお手本の成功エピソードのみ（MuJoCo描画400話、Unreal描画60話。各指示で均等）。")
    T["s5"] = ("評価は、学習に使わない乱数シード（100000〜100009）で各指示10試行、計40試行。")
    T["tab"] = [["方式（学習データ）", "任意", "左", "真ん中", "右", "平均"],
                row("Unreal描画で評価（Unreal 60話＋MuJoCo 400話で学習）", ue),
                row("MuJoCo描画で評価（MuJoCo 400話で学習）", mj)]
    T["s5b"] = [f"写実的なUnreal描画でも、AIは指示された成葉を点群から特定して摘み取れた（平均{pct(ue[1])}、推定誤差の中央値{err_rng}）。",
                f"MuJoCoだけで学習したモデルのUnreal画面での評価は、時間の都合で成葉版では行っていない（MuJoCo画面では{pct(mj[1])}）。",
                f"言い換えた指示（各指示3文×10試行、事前登録して固定）での成功率は、Unreal＋MuJoCoで学習したモデルをMuJoCo描画で評価して{pct(pp)}（同じ条件で元の文は{pct(po)}、相対{pdrop:.0f}%の低下）。指示文は4種類に固定で、言い換えには対応していない。固定の4文で学習し、言い換えは評価だけに使った。",
                "動画の4試行（各指示1試行、seed 100003、100001、100002、100004）は結果を見る前に決めた代表例で、成功率は上の40試行の評価による。動画は10fpsのシミュレーションを10fpsで書き出した等速で、倍速部分はない（タイトルと終わりの画面は静止画）。"]
    T["fig4"] = "図4　成葉の位置推定＋IKの成功率（指示ごと、各10試行）。"
    T["fig5"] = "図5　動画の一場面。全景、アーチ上のカメラ、手首カメラ、点群とAIの推定把持点（黄色の丸）。"
    T["s6"] = ["前の版（同じリポジトリの main ブランチ、v7）は、天井吊りの2関節アームと球殻グリッパーで、決まった位置の葉を順に回収するもので、AIは使っていなかった。本版では、言葉で指定した成葉をAIがカメラの点群から探して摘む形に発展させた。",
               "物理はMuJoCo、描画はUnreal Engineで分担し、MuJoCoで計算した全物体とカメラの姿勢を毎ステップUnrealへ送って、画面のないGPUサーバーで写実的なRGB画像と深度を得る仕組みを作った。UnrealRoboticsLab（MuJoCoとUnrealの連携）の考え方を参考にしたが、そのままでは画面なし環境でカメラ画像が出ず、複数起動で停止したため、姿勢を送る方式に置き換えた。",
               "茶樹・パイプハウス・地面は、実際の茶樹の写真を参考にプログラムで生成した。外部の3Dアセットは使っていない。",
               "方式の選定：開発の途中（株の先端の芽を摘む課題としていた段階）で、動作までAIが出すエンドツーエンド型（点群VLA、SmolVLA）と比べ、少数のデモでは「どれか・どこか」をAIが推定し動作は運動学で計画する役割分担型が最も確実だったため（Unreal描画で役割分担型97.5%、点群VLA 75%、MuJoCo描画でSmolVLA 5%）、成葉版では役割分担型を採用した。"]
    T["s7"] = "物理シミュレーション MuJoCo 3.17.0、描画 Unreal Engine 5.7.4（Lumen、画面なしのVulkan）、学習 PyTorch 2.10.0、指示文の特徴 SmolVLM2-500M-Video-Instruct、比較用VLA LeRobot 0.4.4 のSmolVLA。アームはAdeept ADA031（5自由度、2本指グリッパー）のシミュレーションモデル。"
    T["s8"] = ["成葉は薄く軽いため、離したあとグリッパーに挟まったまま残ることがあった。葉の当たり判定を短くし、葉身の下半分を挟むようにして、お手本の成功率を100試行中98回にした。",
               "掴んだ葉はグリッパーの中心からずれてぶら下がるため、かごへの落とし位置を、指を閉じた時点の推定点とグリッパー先端の差で補正した。",
               "デモの記録を成功判定の瞬間で打ち切ると、グリッパーを開き切る動作が学習データに入らず、方策が「離す直前」で止まった（開発途中の点群VLAで発見）。記録をお手本の最後まで延ばして解決した。",
               "言い換えに弱い理由：学習に使った4文は位置を表す1語だけが違い、言語モデルの文全体の埋め込みでは位置の違いがごく小さな差にしか表れないため、言い換えるとその差が崩れて目標の位置を取り違える。"]
    T["s9"] = ["成葉は剛体の模型で、切断・柔らかさ・損傷は扱わない。成葉は株の上部に立っており、遮蔽は少ない。",
               "実写の茶葉での認識は未実施。実機（Adeept）への移行には、カメラ校正と実測点群での追加学習が必要。",
               "指示文は4種類に固定。改善の方向：学習のときに言い換えの文を混ぜて位置の語の意味を学ばせる、文全体を平均した埋め込みの代わりに語ごとの埋め込みなど位置の語が埋もれにくい表し方にする。",
               "摘む／残すの判断は指示文の位置指定で代用しており、収穫に適した葉そのものの識別は今後の課題。"]
    T["s10"] = ["MuJoCo（Google DeepMind、Apache-2.0）、Unreal Engine 5.7（Epic Games、Unreal Engine EULA）",
                "UnrealRoboticsLab：MuJoCoとUnreal Engineを連携させる考え方を参考にした（本リポジトリにはコードを含めていない）",
                "LeRobot と SmolVLA、SmolVLM2（Hugging Face、Apache-2.0）。重みはリポジトリに含めていない",
                "Adeept ADA031 のシミュレーションモデルと逆運動学：本プロジェクト用に作成し、adeept_sim/ に同梱（キットの説明書の寸法から作成）",
                "茶樹の参照写真：形と色を合わせるための参考に使った。学習データには使っておらず、リポジトリにも含めない",
                "日本語フォント：BIZ UDゴシック（SIL Open Font License）"]

    # ---------- PDF
    st = lambda size=10.5, lead=None: ParagraphStyle("s", fontName="JP", fontSize=size, leading=lead or size * 1.6)
    H1, H2, P = st(17), st(13), st(10.2)
    H2.spaceBefore = 10.2 * 1.6
    S = []
    add = lambda t, s=P: S.append(Paragraph(t, s))
    img = lambda f, w: S.append(RLImage(str(FIG / f), width=w, height=w * Image.open(FIG / f).size[1] / Image.open(FIG / f).size[0]))
    bullets = lambda xs: [add("・" + x) for x in xs]
    add(T["title"], H1); add(T["sub"], st(9.5))
    if links:
        add("　".join(f"{k}：{u}" for k, u in links), st(9.5))
    S.append(Spacer(1, 4 * mm))
    add("1. 課題の概要と目的", H2); add(T["s1"]); img("fig_greenhouse.jpg", 165 * mm); add(T["fig1"], st(9))
    add("2. タスク設定と成功条件", H2); bullets(T["s2"])
    add("3. システム構成と処理の流れ", H2); img("fig_system.png", 170 * mm); add(T["fig2"], st(9)); img("fig_cameras.jpg", 165 * mm); add(T["fig3"], st(9))
    S.append(PageBreak())
    add("4. AIを活用した箇所と、その必要性", H2); add(T["s4"]); add(T["s4b"])
    add("5. 結果", H2); add(T["s5"])
    cell = st(8.8)
    t = Table([[Paragraph(c, cell) for c in r] for r in T["tab"]], colWidths=[70 * mm] + [20 * mm] * 5)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "JP", 8.8), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8efe4"))]))
    S.append(t); S.append(Spacer(1, 3 * mm))
    img("fig_results.png", 170 * mm); add(T["fig4"], st(9)); bullets(T["s5b"])
    img("fig_video_frame.jpg", 160 * mm); add(T["fig5"], st(9))
    add("6. 工夫した点と、前の版から発展させた点", H2); bullets(T["s6"])
    add("7. 実装と使用した主な技術", H2); add(T["s7"])
    add("8. 失敗例と考察", H2); bullets(T["s8"])
    add("9. 制約と今後", H2); bullets(T["s9"])
    add("10. 参考にしたもの・使用した資産", H2); bullets(T["s10"])
    SimpleDocTemplate(str(ROOT / "docs" / "report" / "report.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm).build(S)

    # ---------- Markdown (same text)
    md = [f"# {T['title']}", "", T["sub"] + " PDF版は `scripts/make_report_pdf.py` で作る（`docs/report/report.pdf`）。", ""]
    if links:
        md += ["　".join(f"{k}：{u}" for k, u in links), ""]
    sec = lambda h: md.extend([f"## {h}", ""])
    li = lambda xs: md.extend([f"- {x}" for x in xs] + [""])
    sec("1. 課題の概要と目的"); md += [T["s1"], ""]
    sec("2. タスク設定と成功条件"); li(T["s2"])
    sec("3. システム構成と処理の流れ")
    md += ["```text", "アーチ上のRGB-Dカメラ4台＋手首カメラ（Unreal Engine 5.7で描画、深度つき）", "        ↓ 各カメラの外部パラメータで台座標へ逆投影・統合（2048点、xyz+色）",
           "AI：点群＋指示文 → 指定された成葉の3D把持点", "        ↓", "逆運動学でアームの関節角を計画 → 接近 → 把持 → 引き抜き → かごへ移動 → 解放", "        ↓", "成功判定・記録", "```", ""]
    sec("4. AIを活用した箇所と、その必要性"); md += [T["s4"], "", T["s4b"], ""]
    sec("5. 結果"); md += [T["s5"], "", "| " + " | ".join(T["tab"][0]) + " |", "|" + "---|" * len(T["tab"][0])]
    md += ["| " + " | ".join(r) + " |" for r in T["tab"][1:]] + [""]
    li(T["s5b"])
    sec("6. 工夫した点と、前の版から発展させた点"); li(T["s6"])
    sec("7. 実装と使用した主な技術"); md += [T["s7"], ""]
    sec("8. 失敗例と考察"); li(T["s8"])
    sec("9. 制約と今後"); li(T["s9"])
    sec("10. 参考にしたもの・使用した資産"); li(T["s10"])
    (ROOT / "docs" / "report" / "report.md").write_text("\n".join(md))
    print("pdf", ROOT / "docs" / "report" / "report.pdf", "md", ROOT / "docs" / "report" / "report.md")


if __name__ == "__main__":
    main()
