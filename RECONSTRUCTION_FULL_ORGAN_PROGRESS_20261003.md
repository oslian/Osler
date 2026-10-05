# E25進度：完整對照與獨立核對已完成

目前結論見[完成報告](RECONSTRUCTION_FULL_ORGAN_FINDINGS_20261003.md)。
完整增量screen為0，不promote；本頁保存操作進度，不將工程完成當作準度修復。
科學契約維持[原protocol](RECONSTRUCTION_FULL_ORGAN_PROTOCOL_20261003.md)，未調門檻。

## 全範圍完成

| Fold | Fit查詢 | Selection查詢 | 狀態 |
|---|---:|---:|---|
| 0 | 698,766 | 363,526 | 完整 |
| 1 | 698,510 | 363,526 | 完整 |
| 2 | 698,766 | 356,944 | 完整 |
| 3 | 682,000 | 356,944 | 完整 |
| 4 | 743,818 | 356,944 | 完整 |
| 合計 | 3,521,860 | 1,797,884 | 5,319,744次，10/10角色 |

53指標完整盤點、47支持；四個新HGB100臂×235組=940讀取器。
原235個raw讀取器的資料、權重與預測契約通過才重用，不默默refit。
保存模型／fit和selection評分均完成；獨立核對1,175個selection reader重播與統計通過。
全量cache核對包含key、payload、source/hash、native parity、pilot prefix；
器官state的獨立逐事件演化回放仍只首批兩角色各512筆，不冒充全量重播。
41項相關回歸通過。Runtime權重、registry、區間及serving均未改。

## 已完成的操作歷史

| 階段 | 保存查詢／完整角色 | 程序與結果 |
|---|---|---|
| Pilot及小批續跑 | 17,408／0 | audit87662、提取71813/75812/5948/11225、獨立回放1075皆exit0 |
| 原fold0有限續跑 | 117,408／0 | 提取48626、audit6968皆exit0；37項測試5021通過，partial拒絕41600 |
| Fold0 selection完整 | 472,230／1 | 提取42863、audit31513皆exit0；40項測試28277，缺assessment拒絕40563 |
| Fold0完整 | 1,062,292／2 | 提取13936、audit74015皆exit0；其餘角色未完成時32032拒絕fit |
| Fold1完整 | 2,124,328／4 | fit47987、selection89627、audit13078皆exit0 |
| Fold2完整 | 3,180,038／6 | fit64671、selection70567、audit5760皆exit0 |
| Fold3完整 | 4,218,982／8 | fit73148、selection32131、audit12363皆exit0 |
| Fold4及全量完整 | 5,319,744／10 | fit18974、selection35294、audit35577皆exit0 |
| 全五fold讀取器 | 235組／940新模型 | 訓練29934正常exit0；235原raw核對後重用 |
| 正式評分 | 47指標／29,580病人 | score5765正常exit0，完整增量與全部對照交集皆0 |
| 獨立核對首次 | 模型重現，統計拒絕 | 16965 exit1；32個p值roundoff超64eps，不寫verification |
| 數值修正與重驗 | 1,175reader／235組 | math.fsum修正，41tests36422通過；83993 exit0，最大差2.9976e-15 |

早期partial完整性拒絕、合成端到端測試只證明工程防線，不是病人準度。
初次3項fixture錯誤（合成variables缺map）已修，真實提取無此錯誤。
獨立verifier的32個p值差異最大7.7938e-14，源自逐筆累加；沒有screen差異。
改用高精度加總後重跑全部核對，未放寬64eps或任何科學門檻。
原assessment不覆寫；verification綁定其hash，表示獨立核對完成。

## 成本紀錄與安全邊界

Fold0 fit新增590062耗時8014.44s、selection新增354822耗時4939.46s；
fold1 fit/selection為9312.34s/5115.47s，fold2為9382.16s/4932.18s，
fold3為9053.35s/4844.46s，fold4為10221.61s/4875.26s。
以上包含啟動、hash與快取核對，不能以小pilot線性推成可靠工期上限。
全提取完成快照可用13,516,800,000bytes，運行保留10GiB；未刪既有資料。
所有本輪程序已結束，不聲稱背景繼續。完整根因／泛化修法仍未交付。

## 可重現入口

以下是完成紀錄中的入口，不是要求覆寫既有產物或重新訓練：

```sh
python -m scripts.cache_full_organ_compartments --fold 0 --role fit --max-new 100000
python -m scripts.audit_full_organ_progress
python -m scripts.train_full_organ_readout --fold 0
# 原實驗同樣完成fold1至4，再評分與核對。
python -m scripts.score_full_organ_readout
python -m scripts.verify_full_organ_readout
```

`--max-new`只控制單次續跑量，不改範圍。十份cache不完整即拒絕fit；
source/hash/role違規停止，不刪contract繞過。原始逐筆資料、SQLite、
joblib和parquet留在git-ignore的私有目錄，公開證據僅aggregate。

## 證據入口

- [全量快照](artifacts/native_reconstruction_full_organs_progress_5319744_20261003.json)
- [正式評分](artifacts/native_reconstruction_full_organ_readout_assessment_20261003.json)
- [獨立核對](artifacts/native_reconstruction_full_organ_readout_verification_20261003.json)
- [Pilot回放](artifacts/native_reconstruction_full_organs_pilot_verification_20261003.json)
- 歷史快照：`artifacts/native_reconstruction_full_organs_progress_{17408,117408,472230,1062292,2124328,3180038,4218982}_20261003.json`。
- 提取報告：`artifacts/native_reconstruction_full_organs_fold{0..4}_{fit,selection}_20261003.json`。
- 讀取器報告：`artifacts/native_reconstruction_full_organ_readers_fold{0..4}_20261003.json`。
- 程式：`cache_full_organ_compartments`、`audit_full_organ_progress`、`train_full_organ_readout`、`score_full_organ_readout`、`verify_full_organ_readout`（均在`scripts/`）。
- 測試：`tests/test_full_organ_{cache,progress,readout,score,verification}.py`。
