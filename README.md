# 球殻型・選択摘採グリッパー

**最終課題は下の adeept-urlab ブランチ（点群AIによる茶の成葉の選択摘採）です。** 以下のv7は途中段階の試作です。

Physical AI応用1 最終課題に向けた、葉を1枚ずつ包んで切り離し、球の中に保持して回収するロボットの試作です。

現在は **v7：同じ株の複数の葉を、1枚ずつ回収** です。検証配置は3株×各2枚、計6枚。同じ幹から分かれた2枚の対象葉をそれぞれ箱へ運び終えてから次の株へ進みます。50×25×15cmハウスの天井レール・吊り下げ2関節アームで、左から右へ順番に回ります。各回、箱内滞在と空のグリッパーを確認し、古葉・芽・母枝は残します。対象位置は既知の計画巡回です。AI探索、実機、実切断、同期駆動機構は未実装で、最終課題完成版ではありません。

## adeept-urlab ブランチ：点群AIによる茶の成葉の選択摘採（Adeeptアーム＋Unreal Engine）

このブランチは、上記v7（球殻方式）とは別に、実運用構想の実寸パイプハウス（幅5.7m×長さ10m×高さ3.1m）で、天井レール台車から吊った台に載せたAdeept ADA031アームが、言葉で指定した茶の成葉を1枚ずつ摘んでかごへ回収するシミュレーションである（葉柄と枝は株に残る）。物理はMuJoCo 3.17、描画はUnreal Engine 5.7（Lumen、画面なしVulkan）。

- カメラ：ハウスのアーチ（骨）に4台のRGB-Dカメラ＋手首カメラ。深度から台座標の点群（2048点、xyz＋色）を統合。
- AI：点群と指示文（任意／左／真ん中／右の4種類で固定）から目標の成葉の3D把持点を推定（`scripts/pcvla/target_net.py`）。動作はAIの推定点だけを使ってアームの逆運動学で計画。
- 結果（学習に使わない乱数シードで各指示10試行）：Unreal描画 92.5%、MuJoCo描画 95.0%。言い換えた指示では、Unreal＋MuJoCoで学習したモデルをMuJoCo描画で評価して 63.3%（同じ条件で元の文は 87.5%）で、言い換えには対応していない。詳細は `docs/report/report.md`、`docs/experiment-log.md`。
- 動画：`docs/preview/final/final_video_leaf_fixed.mp4`（カメラ固定）、`final_video_leaf_slow.mp4`（ゆっくり回る視点）。約72秒、等速、1080p（プレビューは720p）。

最小の実行手順（Linux、GPU、Unreal Engine 5.7とプロジェクトの準備は `docs/experiment-log.md` 参照）：

```sh
python scripts/tea_plant_gen.py && python scripts/ue_env_gen.py      # 茶樹・環境メッシュ（手続き生成）
# Unreal: scripts/ue/build_tea_scene.py でレベル構築、scripts/ue/render_server.py を起動
python scripts/make_tea_demos.py --episodes 25 --pointcloud --port 5600 # お手本デモ（Unreal描画）
python scripts/make_tea_demos.py --episodes 100 --pointcloud --renderer mujoco --root output/adeept/datasets/tea_mj3  # MuJoCo描画版
python scripts/pcvla/embed_language.py                                # 指示文の埋め込み（SmolVLM2）
python scripts/pcvla/target_net.py train --data <点群データのフォルダ,...>
python scripts/pcvla/target_net.py eval --renderer ue --ckpt <ckpt>   # 評価
python scripts/make_final_video.py --ckpt <ckpt>                      # 動画
```

Adeeptアームのモデル・運動学・お手本動作は `adeept_sim/` に同梱している（Adeept ADA031 V4 キットの説明書の寸法から作ったMuJoCoモデル。`ADEEPT_SIM` で別の場所も指定できる）。茶樹・ハウス・地面などの見た目は写真を参考にプログラムで生成し、外部アセットは使っていない。

## 生成されるもの

- OnshapeやFreeCADへ取り込めるSTEPアセンブリと部品
- 球殻1枚、ベース、鈍い切断部代用形状のSTL
- 同じ寸法から作ったMuJoCoモデル（`scene.xml`）
- 実際の物理計算の軌跡を視点変更・時刻指定できる3D再生画面
- 捕獲・収納の試験結果と動画

寸法の元データは `config/cage.json` です。CADの単位はmm、シミュレーションはm・kg・radです。CADとシミュレーションはこの設定から生成し、片方だけを手修正しないでください。

## セットアップ

Python 3.12とuvを使用します。macOS Intel、Python 3.12、CadQuery 2.5.2、MuJoCo 3.2.7で検証しました。別環境での依存関係の導入・描画は未検証です。

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 標準の生成・検証手順

リポジトリルートから実行します。

```sh
.venv/bin/python scripts/build_cage.py
.venv/bin/python scripts/check_cage_geometry.py
.venv/bin/python scripts/test_arm_kinematics.py
.venv/bin/python scripts/simulate_cage.py --patrol --test
.venv/bin/python scripts/simulate_cage.py --patrol --video
.venv/bin/python scripts/test_patrol.py
.venv/bin/python scripts/package_cage.py --patrol
```

`output/cage/preview.html` をブラウザで開くと記録を再生できます。3D表示ライブラリの読込みにはインターネット接続が必要です。ローカルサーバーを使う場合：

```sh
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory output/cage
```

ブラウザで `http://127.0.0.1:8765/preview.html` を開いてください。再生画面は保存済みのMuJoCo計算結果であり、ブラウザ内で物理計算を再実行するものではありません。

MuJoCoのウィンドウで実行する場合、macOSでは次を使います。

```sh
.venv/bin/mjpython scripts/simulate_cage.py --patrol --viewer
```

その他の対応環境では `.venv/bin/python scripts/simulate_cage.py --patrol --viewer` で起動します。巡回動画は `patrol.mp4`、モデルは `scene-patrol.xml`、判定は `patrol-report.json` と `patrol-test-report.json` です。`--patrol` を付けない単発動作・試験も比較用に残しています。

## Onshapeで編集する

ハウス全体は `output/cage/step/greenhouse_assembly.step`、台車以下は `output/cage/step/arm_assembly.step`、グリッパー単体は `output/cage/step/cage_assembly.step` をインポートしてください。STEPは初期の閉じた姿勢の概念CADです。フィーチャー履歴、Mate、制御情報は含まれません。

Onshapeを以後の設計元にする場合は、6枚の球殻に回転Mateを設定し、`onshape-to-robot` 用の関節名・質量・衝突形状を整備する必要があります。現在のMuJoCoモデルは、インポートやアカウント登録を待たずに実行できます。

## 検証の意味と限界

v7の巡回は3株の6枚を連続して扱います。収納を確認する条件は「箱内へ1秒滞在し、グリッパーが空になったこと」。完全静止とは別に判定・記録し、株別の回収数、最大同時搬送枚数、意図しない接触、切断・収納順序も記録します。途中の異常は停止し、その後の葉を切りません。

最新の巡回試験結果は `output/cage/patrol-test-report.json` と実験記録を参照してください。通常・半刻み・Y+1mmの3条件と、1株目の2枚目への古葉侵入・開放の対照で確認します。既知の限定配置での結果で、実機の収穫成功率ではありません。

v7の通常条件は6/6枚、約131.5秒で回収し、古葉侵入対照では1枚回収後に停止しました。追加試験一式は長時間化のため途中で中断し、半刻み・Y+1mm・開放対照は結果未確定です。`patrol-test-report.json` は未完了を示し、全条件合格ではありません。

旧v5の単発試験は10条件中9条件が完全静止まで合格で、半刻み・Y+1mmの1条件は箱内で揺れが残り未合格でした。この結果は変更せず記録に残しています。巡回の箱内滞在判定をもって旧試験全合格とはしません。詳細は実験記録を参照してください。

- 対象：葉身1枚（18×6×2mm）と短い葉柄、計35mgの模型剛体。芽や2枚目の葉は含みません。寸法・質量は仮定で、写真からの実測ではありません。茶葉の柔軟性はモデル化していません。
- 残す部分：巡回では各株の共通の母枝、二又の支持枝、古葉2枚、隣の芽。固定された接触対象として配置し、接触で中止します。固定のため、変形・引きちぎり・損傷を検証したことにはなりません。旧単発モデルの植物配置とは異なります。
- 位置合わせ：対象と切断点はあらかじめ既知。XYの位置ずれが1.25mmを超えた場合は切断しません。この判定はAIやカメラの出力ではありません。
- 切断：仮想の刃が閉じた後、対象の葉柄の固定拘束だけを解除します。モデル構造の検査で「葉1枚＋葉柄」以外が解除対象に含まれないことを確認します。破断力や切断品質の検証ではありません。
- 捕獲：解除後は物理接触と重力のみで保持します。対象を球殻へ接着・固定しません。
- 接触：葉の転がり・ねじれ摩擦も計算します（MuJoCo condim=6、摩擦係数は仮定）。実葉の摩擦係数や柔軟性を測定したものではありません。次へ進む判定にシミュレータの真値を使用し、実センサーは未実装です。
- 動作：株中心X=−40、40、120mm、各株の対象葉は中心からX±20mm。葉の巡回順は−60、−20、20、60、100、140mm、箱中心X=230mm。上腕・前腕各60mm。巡回移動は肩−8°/肘16°へ腕を上側に畳み、台車を動かします。XZ平面だけを動き、手首は固定。球殻6枚は理想化した独立駆動です。実サーボや配線・球殻の同期機構は未設計です。
- 中空の球を1つの凸形状にすると中身まで埋まるため、当たり判定は小さな凸パッチに分割しています。
- 球殻同士・ベースとの衝突は動的計算から除外し、CADの静的干渉チェックで補完します。対象の葉と固定植物の相互接触も簡略化しています。
- 旧単発試験は中央・XY各±1mmの5位置を通常刻みと半分の刻みで確認。巡回試験は通常・半刻み・Y+1mmと、開放・途中の古葉侵入の対照を確認します。両試験の条件・収納基準は異なります。限定条件のモデル試験であり、収穫成功率やAI精度ではありません。
- 閉じた球殻外径32mm。ハウスの骨組みを接触対象に含めます。透明カバーは表示のみ。実モーターの収まり、車輪、駆動機構、屋根の荷重・強度は未検証です。

詳しい形状、未設計部分、次の検証項目は `docs/arm-design.md` と `docs/cage-design.md`、実験記録は `docs/experiment-log.md` を参照してください。

## ファイル配置

| 場所 | 内容 |
|---|---|
| `config/cage.json` | 共通寸法と計算設定 |
| `scripts/build_cage.py` | CAD・STL・MuJoCo生成 |
| `scripts/arm_kinematics.py` | 2関節の順運動学・逆運動学・可動範囲チェック |
| `scripts/simulate_cage.py` | 物理試験・動画・動作記録 |
| `scripts/check_cage_geometry.py` | STLの閉じた形状・CADの干渉検査 |
| `scripts/package_cage.py` | 3D再生画面・配布Zipの生成 |
| `output/cage/` | 生成物。Git管理対象外 |
| `docs/references.md` | 外部資料の出典 |

コードは本プロジェクト向けに作成しました。外部のグリッパーCADや記事画像は同梱していません。依存ライブラリの利用条件は各プロジェクトのライセンスに従ってください。写真の出典・利用条件は `data/README.md` を参照してください。

## 利用について

このリポジトリは Physical AI応用1 最終課題の提出用です。ライセンスは付けていないため、閲覧以外の再利用（複製・改変・再配布）は許可していません。
