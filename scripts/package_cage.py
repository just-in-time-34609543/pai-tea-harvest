"""Create a standalone recorded-physics preview and portable design archive."""
from pathlib import Path
import zipfile
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/cage"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--patrol",action="store_true")
    args=parser.parse_args()
    template=(ROOT/"scripts/preview.template.html").read_text()
    suffix="-patrol" if args.patrol else ""
    for token,filename in [("__GEOMETRY__",f"geometry{suffix}.json"),("__TRAJECTORY__",f"trajectory{suffix}.json")]:
        template=template.replace(token,(OUT/filename).read_text())
    (OUT/"preview.html").write_text(template)
    with zipfile.ZipFile(OUT/"cage-design.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for folder in ["step","stl","meshes"]:
            for path in sorted((OUT/folder).glob("*")):
                archive.write(path,path.relative_to(OUT))
        for name in ["scene.xml","scene-patrol.xml","build-report.json","run-report.json","test-report.json","patrol-report.json","patrol-test-report.json","patrol-test-report-v6.json","geometry-report.json","trajectory.json","trajectory-patrol.json","measurements.json","preview.html","demo.mp4","patrol.mp4"]:
            path=OUT/name
            if path.exists(): archive.write(path,name)
        for name in ["config/cage.json","requirements.txt","README.md","docs/arm-design.md","docs/cage-design.md","docs/experiment-log.md","scripts/greenhouse.py","scripts/arm_kinematics.py","scripts/test_arm_kinematics.py","scripts/patrol.py","scripts/build_cage.py","scripts/simulate_cage.py","scripts/check_cage_geometry.py","scripts/package_cage.py","scripts/preview.template.html"]:
            archive.write(ROOT/name, "source/"+name)
        archive.write(ROOT/"scripts/test_patrol.py","source/scripts/test_patrol.py")
        cfg=json.loads((ROOT/"config/cage.json").read_text())
        pc=cfg["patrol"]
        summary=f'{len(pc["plant_centers_mm"])}株×各{len(pc["leaf_offsets_mm"])}枚、同じ株の対象葉を1枚ずつ回収して次の株へ進む動画'
        archive.writestr("はじめに.txt", "葉を1枚ずつ回収する球殻グリッパーの形状試作です。\npreview.html: 計算済み3D再生（ネット接続が必要）。葉を拡大ボタンで対象を確認できます。\npatrol.mp4: "+summary+"\nscene-patrol.xml + meshes/: 巡回用MuJoCoモデル\npatrol-report.json、patrol-test-report.json: 巡回の判定・停止試験\ndemo.mp4、scene.xml、test-report.json: 旧単発モデルの比較記録（v5の未合格条件を含む）\nstep/: Onshape等へ取り込む概念CAD（アーム単発初期姿勢）\nstl/: 形状確認用の印刷データ（駆動機構は未完成）\nsource/: 再生成用ソース。こちらを作業フォルダにしてREADME.mdの手順を実行してください。\n巡回先は事前設定です。実際の切断・柔らかい葉・AI認識・実機動作は未検証です。\n")
    print(OUT/"preview.html")
    print(OUT/"cage-design.zip")

if __name__=="__main__": main()
