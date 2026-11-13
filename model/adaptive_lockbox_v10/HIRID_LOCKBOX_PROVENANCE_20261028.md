# HiRID lockbox: data provenance（protocol v10，2026-10-28）

本檔案列在 HiRID addendum 的檔案中，評分前凍結。

## 決定

2026-10-28，使用者決定以下面這份非官方副本，作為 HiRID 的正式 lockbox 資料，並評分一次。

- HiRID lockbox 依 v10 只能評分一次，此次評分會將它用掉。
- 之後即使取得官方 PhysioNet 檔案，也不能重新評分。

## 來源

| 項目 | 內容 |
|---|---|
| 原始檔 | 第三方轉售的 `hirid-a-high-time-resolution-icu-dataset-1.1.1.rar`，不是從 PhysioNet 下載 |
| 大小 | 7,956,346,146 bytes |
| SHA-256 | `e0759f847c734403981a1fc70d5ae6ab4eb83f5249bd053368b7073a50157771` |
| 本機路徑 | `~/Downloads/hirid/` |

## 可驗證與不可驗證的部分

### `reference_data.tar.gz`：重新打包過，與官方不符

| | SHA-256 |
|---|---|
| 官方 `SHA256SUMS.txt` 記載 | `33670c9eac871e607174a28b9a943c2e2db0f0b885c01eddd9b1ddce39a1f4e5` |
| 這份副本 | `1780432ee6713bf1c0f8c6dff9787d9a0ea9fc9d613c87eba74e3d4aa6fcfd0f` |

- 這份壓縮檔被重新打包，多了 2 個 `.url` 和 1 個 `.jpg` 廣告檔。這些檔案沒有被開啟。
- 四個 CSV 成員的作者欄位與日期（2020–2021）都保留原狀。
- 實際使用的 reference 檔案與壓縮檔內的 CSV 成員逐位元組相同：

| 檔案 | SHA-256 |
|---|---|
| `general_table.csv` | `8d84ecc80cab0406452d28a57957419f2b15e39b82a5719d970a810ef600ecdc` |
| `hirid_variable_reference.csv` | `7f3095dbfe15512c8dd313ce83a6613e18762f0e364b7fbcd0ed177ff9de5f8f` |
| `ordinal_vars_ref.csv` | `c032c8aabaf01a3e88ea88e692d378122926737f931d2ffa1c839392d9218ccc` |

- 因為官方的 checksum 是針對整個 tar 檔，以上比對**無法證明這些 CSV 與官方內容相同**。

### 觀測表：無法以官方 checksum 驗證

- rar 內只有 CSV 版的 `raw_stage/observation_tables/csv/part-*.csv`，共 250 份，而且不是官方的 tar 包裝。
- 這些 CSV 以 duckdb 依 HiRID schemata 的欄位型別無損轉成 parquet，放在官方 parquet 的目錄位置：
  - 每個 part 都比對過 CSV 與 parquet 的列數，結果完全相同；
  - 轉換用的是 session 內的 staging 腳本。
- `observation_tables_index.csv` 的 SHA-256：`bfd855c61f2957a3032aeb8929af7b9f6c58ee0d6fb7f523b36017ea4861090e`。
- 250 份 parquet 的 SHA-256 清單在 `hirid_observation_parquet_manifest_20261028.txt`，與本檔一起記錄在 addendum。
- 官方 `observation_tables_csv.tar.gz` 和 `observation_tables_parquet.tar.gz` 都沒有取得，所以無法比對。

## 對結果解讀的影響

- HiRID 的 lockbox 結果帶有**來源未驗證**的限制：
  - 內容若被竄改、截斷或混入其他資料，無法排除；
  - 支援數與官方文件描述的規模一致，但這只是旁證：33,905 位病人、250 個 part、約 2 分鐘一筆監測資料。
- 回報 HiRID 結果時必須附上本限制，結果的證據強度低於使用經 checksum 驗證官方檔案的結果。
  - SICdb 原始檔的來源也尚未確認：它的 rar 看起來也被重新包裝過，因此同樣不能當作已驗證的對照。
- 使用者另需自行確認此副本的取得方式符合 PhysioNet 的資料使用協議（DUA）。
