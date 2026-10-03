# Phase 1 cleaning log: DataCo Smart Supply Chain

- Run date: 2026-10-03
- Script: `A_preprocessing_eda/phase1_cleaning.py`
- Raw file: `dataset/DataCoSupplyChainDataset.csv`
- Python: 3.13.9
- pandas: 2.3.3
- numpy: 2.3.5

## Summary

The raw file had 180,519 rows and 53 columns. The cleaned file has 180,519 rows and 122 columns. No rows were deleted.
Columns dropped: 4 empty or useless, 5 personal, 9 identifier, 3 late-delivery leakage, 2 profit leakage, 2 exact duplicate (25 in total).
Columns added: 4 derived date, 2 transform, 3 grouped text, 83 one-hot, 2 target (94 in total).
Null cells fell from 336,209 to 0.
Late orders (`late` = 1): 54.8291% (98,977 of 180,519). Loss orders (`loss` = 1): 18.7149% (33,784 of 180,519).
Exactly zero profit: 0.6520% (1,177 rows). All of them have `loss` = 0.
The cleaned file is unscaled, has no train/test split, and is 93.65 MB (98,202,486 bytes). This is above 50 MB. GitHub warns at 50 MB and blocks at 100 MB. Git LFS may be needed. The file was not shrunk.

## Step-by-step log

### Step 1. Load and inspect

The raw file was loaded with `encoding="latin-1"` and was not modified. Shape: 180,519 rows by 53 columns. Deep memory use: 298.92 MB. Duplicate rows: 0. Columns with no nulls: 49 of 53.

Dtype table:

| Column | Dtype |
| --- | --- |
| Type | object |
| Days for shipping (real) | int64 |
| Days for shipment (scheduled) | int64 |
| Benefit per order | float64 |
| Sales per customer | float64 |
| Delivery Status | object |
| Late_delivery_risk | int64 |
| Category Id | int64 |
| Category Name | object |
| Customer City | object |
| Customer Country | object |
| Customer Email | object |
| Customer Fname | object |
| Customer Id | int64 |
| Customer Lname | object |
| Customer Password | object |
| Customer Segment | object |
| Customer State | object |
| Customer Street | object |
| Customer Zipcode | float64 |
| Department Id | int64 |
| Department Name | object |
| Latitude | float64 |
| Longitude | float64 |
| Market | object |
| Order City | object |
| Order Country | object |
| Order Customer Id | int64 |
| order date (DateOrders) | object |
| Order Id | int64 |
| Order Item Cardprod Id | int64 |
| Order Item Discount | float64 |
| Order Item Discount Rate | float64 |
| Order Item Id | int64 |
| Order Item Product Price | float64 |
| Order Item Profit Ratio | float64 |
| Order Item Quantity | int64 |
| Sales | float64 |
| Order Item Total | float64 |
| Order Profit Per Order | float64 |
| Order Region | object |
| Order State | object |
| Order Status | object |
| Order Zipcode | float64 |
| Product Card Id | int64 |
| Product Category Id | int64 |
| Product Description | float64 |
| Product Image | object |
| Product Name | object |
| Product Price | float64 |
| Product Status | int64 |
| shipping date (DateOrders) | object |
| Shipping Mode | object |

Nulls (columns with at least one null):

| Column | Null count | Null percent |
| --- | --- | --- |
| Customer Lname | 8 | 0.0044% |
| Customer Zipcode | 3 | 0.0017% |
| Order Zipcode | 155,679 | 86.2397% |
| Product Description | 180,519 | 100.0000% |

Expected facts:

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Rows | 180,519 | 180,519 | PASS |
| Columns | 53 | 53 | PASS |
| Duplicate rows | 0 | 0 | PASS |
| Late_delivery_risk = 1 | 54.8% (tolerance 0.1 percentage points) | 54.8291% (98,977 rows; difference 0.0291 pp) | PASS |
| Benefit per order < 0 | 18.7% (tolerance 0.1 percentage points) | 18.7149% (33,784 rows; difference 0.0149 pp) | PASS |
| Benefit per order = 0 | 0.7% (tolerance 0.1 percentage points) | 0.6520% (1,177 rows; difference 0.0480 pp) | PASS |
| Nulls in Product Description | 100% (180,519 rows) | 180,519 | PASS |
| Nulls in Order Zipcode | 155,679 | 155,679 | PASS |
| Nulls in Customer Zipcode | 3 | 3 | PASS |
| Nulls in Customer Lname | 8 | 8 | PASS |
| Order date minimum | 2015-01-01 | 2015-01-01 00:00:00 | PASS |
| Order date maximum | 2017-09-09 | 2018-01-31 23:38:00 | DISCREPANCY |

### Step 2. Drop columns

Columns were dropped only from explicit lists. A listed column that is absent is recorded under Discrepancies and does not stop the run. `180,519` rows remained. Column count after this step: 30.

`Days for shipment (scheduled)` was kept. It is the scheduled transit time known when the order is placed, so it is not leakage.

Leakage rules:

- `Delivery Status`: Known only after the delivery outcome is recorded, and it restates whether the order was late.
- `Days for shipping (real)`: The actual days in transit are known only after delivery, and lateness is that value compared with the scheduled days.
- `shipping date (DateOrders)`: The ship timestamp is known only after dispatch, so it is not available at order time and leaks the late-delivery target.
- `Order Profit Per Order`: It defines profit directly. Benefit per order is kept because it is the OLS target and the source of the loss target.
- `Order Item Profit Ratio`: It expresses profit relative to the item value, so it defines the loss and profit targets.

`Benefit per order` was not dropped. It is the continuous profit target and the source of `loss`.

Identifier columns dropped: `Customer Id`, `Order Customer Id`, `Order Id`, `Order Item Id`, `Category Id`, `Department Id`, `Product Card Id`, `Product Category Id`, `Order Item Cardprod Id`. A column was treated as a pure identifier when it is an entity key (customer, order, order line, category, department, or product card). `Latitude` and `Longitude` were kept because they are measurements.

The full drop list, including later exact-duplicate drops, is in the dropped-columns table.

### Step 3. Parse the order date

`order date (DateOrders)` was parsed with `pd.to_datetime` and format `%m/%d/%Y %H:%M`. Failed parses: 0. Nothing was derived from `shipping date (DateOrders)`, which had already been dropped.

Derived columns: `order_year` (calendar year), `order_month` (1-12), `order_weekday` (Monday = 0, Sunday = 6), `order_weekday_name` (fixed English names from that integer, not from the machine locale). The original column is kept as a datetime and written as ISO text `YYYY-MM-DD HH:MM:SS` so Power BI can parse it.

Actual order-date minimum: 2015-01-01 00:00:00. Actual order-date maximum: 2018-01-31 23:38:00. Column count after this step: 34.

### Step 4. Near-duplicates and remaining nulls

Pairs were compared by value with absolute tolerance 1e-09. Identity of `Benefit per order` and `Order Profit Per Order` was measured on the raw file before that column was dropped.

| Left | Right | Differing rows | Max absolute difference | Action |
| --- | --- | --- | --- | --- |
| Benefit per order | Order Profit Per Order | 0 | 0.000000 | Identical before the Step 2 drop. `Order Profit Per Order` was dropped as leakage. `Benefit per order` was kept. |
| Sales per customer | Order Item Total | 0 | 0.000000 | Identical. Dropped `Sales per customer` and kept `Order Item Total`. |
| Order Item Product Price | Product Price | 0 | 0.000000 | Identical. Dropped `Product Price` and kept the other column. |

Systematic scan of the remaining numeric columns: exact duplicates were dropped unless the column is a target, `Benefit per order`, `Sales`, `Order Item Quantity`, `Days for shipment (scheduled)`, `Order Item Total`, `Order Item Product Price`, or a derived date column. When both names were unprotected, the later column in the current frame was dropped. Correlated but non-identical pairs were not dropped.

### High-correlation pairs retained (for C to handle, for example via VIF)

Pearson correlation was computed on the numeric columns that remained after the exact-duplicate drops. Pairs with absolute correlation above 0.95 are listed. Exact duplicates are not repeated here.

| Column A | Column B | Pearson r | Action |
| --- | --- | --- | --- |
| Sales | Order Item Total | 0.989744 | Retained. Not an exact duplicate. |

Null handling: `Customer Zipcode`: 3 missing values filled with `Unknown`. Non-missing values are integral floats and were stored as digit strings without a decimal point. Assertion after this step: null cells = 0. Column count after this step: 32.

### Step 5. Outlier review

Quartiles use `Series.quantile` (linear interpolation). Fences are Q1 - 1.5 IQR and Q3 + 1.5 IQR. Skewness is `Series.skew` (bias-adjusted). No row was deleted or capped. Original columns were not overwritten.

| Column | Q1 | Q3 | IQR | Lower fence | Upper fence | Rows outside | Percent outside | Skewness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sales | 119.980003 | 299.950012 | 179.970009 | -149.975010 | 569.905025 | 488 | 0.2703% | 2.884249 |
| Benefit per order | 7.000000 | 64.800003 | 57.800003 | -79.700005 | 151.500008 | 18,942 | 10.4931% | -4.741834 |
| Order Item Quantity | 1.000000 | 3.000000 | 2.000000 | -2.000000 | 6.000000 | 0 | 0.0000% | 0.880252 |

`Benefit per order` outliers that are also negative (loss rows): 13,727 of 18,942 outlier rows. Those rows stay in the file because they are the loss class.

Observed ranges: `Sales` 9.990000 to 1999.989990; `Benefit per order` -4274.979980 to 911.799988; `Order Item Quantity` 1.000000 to 5.000000.

Transforms:

- `Sales_log` from `Sales`: skewness 2.884249 before, -0.679457 after. It reduced absolute skewness.
- `Benefit_signed_log` from `Benefit per order`: skewness -4.741834 before, -1.444060 after. It reduced absolute skewness.
- `Order Item Quantity` skewness is 0.880252. A log column is added only when absolute skewness is above 1.0, so no log column was added. `log1p` of this column would have skewness 0.630835, which was computed only to record the decision.

A plain log cannot be used on `Benefit per order` because the column contains negative values and zeros. `Benefit_signed_log` keeps the sign, maps 0 to 0, and compresses the magnitude with `log1p`.

Plot files:

- `A_preprocessing_eda/outputs/outlier_plots/box_Sales.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Sales_before.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Sales_after.png`
- `A_preprocessing_eda/outputs/outlier_plots/box_Benefit_per_order.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Benefit_per_order_before.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Benefit_per_order_after.png`
- `A_preprocessing_eda/outputs/outlier_plots/box_Order_Item_Quantity.png`

Column count after this step: 34.

### Step 6. Encode categoricals

Original text columns are kept and encoded columns are added in the same file. Power BI and EDA need readable categories, and the models need numeric columns. Encoded names use the source column as a prefix so they are not confused with the original text. `pd.get_dummies(..., prefix=<column name>, prefix_sep="_", dtype=int)` was used. `drop_first` was not used. Dummy values are integers 0/1, and dummy columns within each group are ordered alphabetically so the file does not depend on row order.

Low-cardinality one-hot columns: `Shipping Mode`, `Market`, `Customer Segment`, `Type`, `Department Name`, `Order Region`. `Order Region` is included because the late-delivery model uses it.

High-cardinality columns use `TOP_N = 10`. The kept values are the most frequent, with ties broken alphabetically. Other values are mapped to `Other`. The grouped text is saved as `<column>_grouped` and then one-hot encoded.

`Shipping Mode`: 4 unique values, all one-hot encoded. 'Standard Class' (107,752), 'Second Class' (35,216), 'First Class' (27,814), 'Same Day' (9,737).

`Market`: 5 unique values, all one-hot encoded. 'LATAM' (51,594), 'Europe' (50,252), 'Pacific Asia' (41,260), 'USCA' (25,799), 'Africa' (11,614).

`Customer Segment`: 3 unique values, all one-hot encoded. 'Consumer' (93,504), 'Corporate' (54,789), 'Home Office' (32,226).

`Type`: 4 unique values, all one-hot encoded. 'DEBIT' (69,295), 'TRANSFER' (49,883), 'PAYMENT' (41,725), 'CASH' (19,616).

`Department Name`: 11 unique values, all one-hot encoded. 'Fan Shop' (66,861), 'Apparel' (48,998), 'Golf' (33,220), 'Footwear' (14,525), 'Outdoors' (9,686), 'Fitness' (2,479), 'Discs Shop' (2,026), 'Technology' (1,465), 'Pet Shop' (492), 'Book Shop' (405), 'Health and Beauty ' (362).

`Order Region`: 23 unique values, all one-hot encoded. 'Central America' (28,341), 'Western Europe' (27,109), 'South America' (14,935), 'Oceania' (10,148), 'Northern Europe' (9,792), 'Southeast Asia' (9,539), 'Southern Europe' (9,431), 'Caribbean' (8,318), 'West of USA ' (7,993), 'South Asia' (7,731), 'Eastern Asia' (7,280), 'East of USA' (6,915), 'West Asia' (6,009), 'US Center ' (5,887), 'South of  USA ' (4,045), 'Eastern Europe' (3,920), 'West Africa' (3,696), 'North Africa' (3,232), 'East Africa' (1,852), 'Central Africa' (1,677), 'Southern Africa' (1,157), 'Canada' (959), 'Central Asia' (553).

`Category Name`: 50 unique values; 10 kept; coverage 88.8278% (160,351 of 180,519 rows). Kept values: 'Cleats' (24,551), "Men's Footwear" (22,246), "Women's Apparel" (21,035), 'Indoor/Outdoor Games' (19,298), 'Fishing' (17,325), 'Water Sports' (15,540), 'Camping & Hiking' (13,729), 'Cardio Equipment' (12,487), 'Shop By Sport' (10,984), 'Electronics' (3,156). Remaining rows mapped to `Other`: 20,168 (11.1722%).

`Order State`: 1,089 unique values; 10 kept; coverage 19.3692% (34,965 of 180,519 rows). Kept values: 'Inglaterra' (6,722), 'California' (4,966), 'Isla de Francia' (4,580), 'Renania del Norte-Westfalia' (3,303), 'San Salvador' (3,055), 'Nueva York' (2,753), 'Distrito Federal' (2,559), 'Texas' (2,446), 'Nueva Gales del Sur' (2,370), 'Santo Domingo' (2,211). Remaining rows mapped to `Other`: 145,554 (80.6308%).

`Order Country`: 164 unique values; 10 kept; coverage 55.4590% (100,114 of 180,519 rows). Kept values: 'Estados Unidos' (24,840), 'Francia' (13,222), 'México' (13,172), 'Alemania' (9,564), 'Australia' (8,497), 'Brasil' (7,987), 'Reino Unido' (7,302), 'China' (5,758), 'Italia' (4,989), 'India' (4,783). Remaining rows mapped to `Other`: 80,405 (44.5410%).

One-hot row-sum checks (each group sums to 1 on every row, values are 0/1, dtype is integer):

- `Shipping Mode`: 4 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Market`: 5 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Customer Segment`: 3 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Type`: 4 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Department Name`: 11 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Order Region`: 23 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Category Name_grouped`: 11 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Order State_grouped`: 11 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.
- `Order Country_grouped`: 11 dummy columns, integer dtype, values in {0, 1}, row sums equal 1 on all 180,519 rows.

New one-hot columns:

- `Shipping Mode_First Class`
- `Shipping Mode_Same Day`
- `Shipping Mode_Second Class`
- `Shipping Mode_Standard Class`
- `Market_Africa`
- `Market_Europe`
- `Market_LATAM`
- `Market_Pacific Asia`
- `Market_USCA`
- `Customer Segment_Consumer`
- `Customer Segment_Corporate`
- `Customer Segment_Home Office`
- `Type_CASH`
- `Type_DEBIT`
- `Type_PAYMENT`
- `Type_TRANSFER`
- `Department Name_Apparel`
- `Department Name_Book Shop`
- `Department Name_Discs Shop`
- `Department Name_Fan Shop`
- `Department Name_Fitness`
- `Department Name_Footwear`
- `Department Name_Golf`
- `Department Name_Health and Beauty `
- `Department Name_Outdoors`
- `Department Name_Pet Shop`
- `Department Name_Technology`
- `Order Region_Canada`
- `Order Region_Caribbean`
- `Order Region_Central Africa`
- `Order Region_Central America`
- `Order Region_Central Asia`
- `Order Region_East Africa`
- `Order Region_East of USA`
- `Order Region_Eastern Asia`
- `Order Region_Eastern Europe`
- `Order Region_North Africa`
- `Order Region_Northern Europe`
- `Order Region_Oceania`
- `Order Region_South America`
- `Order Region_South Asia`
- `Order Region_South of  USA `
- `Order Region_Southeast Asia`
- `Order Region_Southern Africa`
- `Order Region_Southern Europe`
- `Order Region_US Center `
- `Order Region_West Africa`
- `Order Region_West Asia`
- `Order Region_West of USA `
- `Order Region_Western Europe`
- `Category Name_grouped_Camping & Hiking`
- `Category Name_grouped_Cardio Equipment`
- `Category Name_grouped_Cleats`
- `Category Name_grouped_Electronics`
- `Category Name_grouped_Fishing`
- `Category Name_grouped_Indoor/Outdoor Games`
- `Category Name_grouped_Men's Footwear`
- `Category Name_grouped_Other`
- `Category Name_grouped_Shop By Sport`
- `Category Name_grouped_Water Sports`
- `Category Name_grouped_Women's Apparel`
- `Order State_grouped_California`
- `Order State_grouped_Distrito Federal`
- `Order State_grouped_Inglaterra`
- `Order State_grouped_Isla de Francia`
- `Order State_grouped_Nueva Gales del Sur`
- `Order State_grouped_Nueva York`
- `Order State_grouped_Other`
- `Order State_grouped_Renania del Norte-Westfalia`
- `Order State_grouped_San Salvador`
- `Order State_grouped_Santo Domingo`
- `Order State_grouped_Texas`
- `Order Country_grouped_Alemania`
- `Order Country_grouped_Australia`
- `Order Country_grouped_Brasil`
- `Order Country_grouped_China`
- `Order Country_grouped_Estados Unidos`
- `Order Country_grouped_Francia`
- `Order Country_grouped_India`
- `Order Country_grouped_Italia`
- `Order Country_grouped_México`
- `Order Country_grouped_Other`
- `Order Country_grouped_Reino Unido`

Columns left as text and not encoded: `Order Status`, `Product Name`, `Customer City`, `Customer State`, `Customer Country`, `Order City`, `Customer Zipcode`. They are marked in the column dictionary as retained text, not for modelling without a decision. Column count after this step: 120.

### Step 7. Add targets

`late` was copied from `Late_delivery_risk` and both columns were kept. Rows where they differ: 0. `loss` is 1 when `Benefit per order` < 0 and 0 otherwise. Both are integers in {0, 1} with no nulls.

Class counts:

- `late` = 1: 98,977 (54.8291%). `late` = 0: 81,542 (45.1709%).
- `loss` = 1: 33,784 (18.7149%). `loss` = 0: 146,735 (81.2851%).
- `Benefit per order` = 0: 1,177 (0.6520%). Rows with zero profit and `loss` = 0: 1,177. Zero-profit orders are not loss.

Column count after this step: 122.

### Step 8. Save outputs

Columns were ordered as surviving original columns, derived date columns, transforms, grouped text, one-hot columns, then `late` and `loss`. The file was written with `index=False` and `encoding="utf-8"`, then read back. Shape, dtype class, and values matched the in-memory frame (`assert_frame_equal`, float tolerance rtol 1e-7 and atol 1e-9). Integer columns were still integers. `Sales`, `Benefit per order`, and `Order Item Quantity` matched the pre-save values exactly on that re-read.

Final file size is 93.65 MB (98,202,486 bytes). This is above the 50 MB GitHub warning threshold and was not shrunk or truncated. GitHub blocks files at 100 MB. Git LFS may be needed.

Duplicate rows in the cleaned file: 0. Duplicate rows in the raw file: 0. The cleaned row count equals the raw row count. Any increase in duplicate rows is the result of removing identifier columns, not of deleting or copying records.

Seed `42` was set with `numpy.random.seed`. No step in this phase draws random numbers.

## Dropped columns

| Column | Category | Reason |
| --- | --- | --- |
| Product Description | Empty or useless | Entirely null in this file, so it has no information. |
| Order Zipcode | Empty or useless | Almost entirely null in this file, so it is not a usable feature. |
| Product Status | Empty or useless | Listed as empty or useless. Distinct values in this file: 1. |
| Product Image | Empty or useless | Image URL, not used by the models or the hypothesis tests. |
| Customer Password | Personal | Personal data. Removed so the cleaned file can be shared inside the group. |
| Customer Email | Personal | Personal data. Removed so the cleaned file can be shared inside the group. |
| Customer Fname | Personal | Personal data. Removed so the cleaned file can be shared inside the group. |
| Customer Lname | Personal | Personal data. Removed so the cleaned file can be shared inside the group. |
| Customer Street | Personal | Personal data. Removed so the cleaned file can be shared inside the group. |
| Customer Id | ID columns | Customer key. It identifies a person and is not a descriptive feature. Matches `Order Customer Id` on every row (maximum absolute difference 0.000000). |
| Order Customer Id | ID columns | Customer key stored on the order. It is not a descriptive feature. Matches `Customer Id` on every row (maximum absolute difference 0.000000). |
| Order Id | ID columns | Order key. It is not a descriptive feature. |
| Order Item Id | ID columns | Order-line key. It is not a descriptive feature. |
| Category Id | ID columns | Category key. The category name is kept instead. Matches `Product Category Id` on every row (maximum absolute difference 0.000000). |
| Department Id | ID columns | Department key. The department name is kept instead. |
| Product Card Id | ID columns | Product key. The product name is kept instead. Matches `Order Item Cardprod Id` on every row (maximum absolute difference 0.000000). |
| Product Category Id | ID columns | Category key. The category name is kept instead. Matches `Category Id` on every row (maximum absolute difference 0.000000). |
| Order Item Cardprod Id | ID columns | Product key stored on the order line. The product name is kept instead. Matches `Product Card Id` on every row (maximum absolute difference 0.000000). |
| Delivery Status | Leakage for the late-delivery model | Known only after the delivery outcome is recorded, and it restates whether the order was late. |
| Days for shipping (real) | Leakage for the late-delivery model | The actual days in transit are known only after delivery, and lateness is that value compared with the scheduled days. |
| shipping date (DateOrders) | Leakage for the late-delivery model | The ship timestamp is known only after dispatch, so it is not available at order time and leaks the late-delivery target. |
| Order Profit Per Order | Leakage for the loss and profit models | It defines profit directly. Benefit per order is kept because it is the OLS target and the source of the loss target. Before the drop, differing rows versus `Benefit per order`: 0; maximum absolute difference 0.000000. |
| Order Item Profit Ratio | Leakage for the loss and profit models | It expresses profit relative to the item value, so it defines the loss and profit targets. |
| Sales per customer | Exact duplicate | Identical to `Order Item Total` (0 differing rows, maximum absolute difference 0.000000, tolerance 1e-09). `Order Item Total` was kept. |
| Product Price | Exact duplicate | Exact duplicate of `Order Item Product Price` (0 differing rows, maximum absolute difference 0.000000, tolerance 1e-09). `Order Item Product Price` was kept because it appears earlier or is protected. |

## Added columns

| Column | Type | How derived |
| --- | --- | --- |
| order_year | derived date (int64) | Calendar year of `order date (DateOrders)`. |
| order_month | derived date (int64) | Calendar month of `order date (DateOrders)`, numbered 1 to 12. |
| order_weekday | derived date (int64) | Weekday of `order date (DateOrders)`. Monday is 0 and Sunday is 6. |
| order_weekday_name | derived date (text) | English weekday name mapped from `order_weekday` with a fixed Monday-first list. |
| Sales_log | transform (float64) | `log1p(Sales)`. `Sales` was asserted non-negative. The original column is unchanged. |
| Benefit_signed_log | transform (float64) | `sign(Benefit per order) * log1p(abs(Benefit per order))`. The original column is unchanged. |
| Shipping Mode_First Class | one-hot (int64) | Integer 0/1 indicator for `Shipping Mode` = 'First Class'. Every level of this group is kept. |
| Shipping Mode_Same Day | one-hot (int64) | Integer 0/1 indicator for `Shipping Mode` = 'Same Day'. Every level of this group is kept. |
| Shipping Mode_Second Class | one-hot (int64) | Integer 0/1 indicator for `Shipping Mode` = 'Second Class'. Every level of this group is kept. |
| Shipping Mode_Standard Class | one-hot (int64) | Integer 0/1 indicator for `Shipping Mode` = 'Standard Class'. Every level of this group is kept. |
| Market_Africa | one-hot (int64) | Integer 0/1 indicator for `Market` = 'Africa'. Every level of this group is kept. |
| Market_Europe | one-hot (int64) | Integer 0/1 indicator for `Market` = 'Europe'. Every level of this group is kept. |
| Market_LATAM | one-hot (int64) | Integer 0/1 indicator for `Market` = 'LATAM'. Every level of this group is kept. |
| Market_Pacific Asia | one-hot (int64) | Integer 0/1 indicator for `Market` = 'Pacific Asia'. Every level of this group is kept. |
| Market_USCA | one-hot (int64) | Integer 0/1 indicator for `Market` = 'USCA'. Every level of this group is kept. |
| Customer Segment_Consumer | one-hot (int64) | Integer 0/1 indicator for `Customer Segment` = 'Consumer'. Every level of this group is kept. |
| Customer Segment_Corporate | one-hot (int64) | Integer 0/1 indicator for `Customer Segment` = 'Corporate'. Every level of this group is kept. |
| Customer Segment_Home Office | one-hot (int64) | Integer 0/1 indicator for `Customer Segment` = 'Home Office'. Every level of this group is kept. |
| Type_CASH | one-hot (int64) | Integer 0/1 indicator for `Type` = 'CASH'. Every level of this group is kept. |
| Type_DEBIT | one-hot (int64) | Integer 0/1 indicator for `Type` = 'DEBIT'. Every level of this group is kept. |
| Type_PAYMENT | one-hot (int64) | Integer 0/1 indicator for `Type` = 'PAYMENT'. Every level of this group is kept. |
| Type_TRANSFER | one-hot (int64) | Integer 0/1 indicator for `Type` = 'TRANSFER'. Every level of this group is kept. |
| Department Name_Apparel | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Apparel'. Every level of this group is kept. |
| Department Name_Book Shop | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Book Shop'. Every level of this group is kept. |
| Department Name_Discs Shop | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Discs Shop'. Every level of this group is kept. |
| Department Name_Fan Shop | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Fan Shop'. Every level of this group is kept. |
| Department Name_Fitness | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Fitness'. Every level of this group is kept. |
| Department Name_Footwear | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Footwear'. Every level of this group is kept. |
| Department Name_Golf | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Golf'. Every level of this group is kept. |
| Department Name_Health and Beauty  | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Health and Beauty '. Every level of this group is kept. |
| Department Name_Outdoors | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Outdoors'. Every level of this group is kept. |
| Department Name_Pet Shop | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Pet Shop'. Every level of this group is kept. |
| Department Name_Technology | one-hot (int64) | Integer 0/1 indicator for `Department Name` = 'Technology'. Every level of this group is kept. |
| Order Region_Canada | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Canada'. Every level of this group is kept. |
| Order Region_Caribbean | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Caribbean'. Every level of this group is kept. |
| Order Region_Central Africa | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Central Africa'. Every level of this group is kept. |
| Order Region_Central America | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Central America'. Every level of this group is kept. |
| Order Region_Central Asia | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Central Asia'. Every level of this group is kept. |
| Order Region_East Africa | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'East Africa'. Every level of this group is kept. |
| Order Region_East of USA | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'East of USA'. Every level of this group is kept. |
| Order Region_Eastern Asia | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Eastern Asia'. Every level of this group is kept. |
| Order Region_Eastern Europe | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Eastern Europe'. Every level of this group is kept. |
| Order Region_North Africa | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'North Africa'. Every level of this group is kept. |
| Order Region_Northern Europe | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Northern Europe'. Every level of this group is kept. |
| Order Region_Oceania | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Oceania'. Every level of this group is kept. |
| Order Region_South America | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'South America'. Every level of this group is kept. |
| Order Region_South Asia | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'South Asia'. Every level of this group is kept. |
| Order Region_South of  USA  | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'South of  USA '. Every level of this group is kept. |
| Order Region_Southeast Asia | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Southeast Asia'. Every level of this group is kept. |
| Order Region_Southern Africa | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Southern Africa'. Every level of this group is kept. |
| Order Region_Southern Europe | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Southern Europe'. Every level of this group is kept. |
| Order Region_US Center  | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'US Center '. Every level of this group is kept. |
| Order Region_West Africa | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'West Africa'. Every level of this group is kept. |
| Order Region_West Asia | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'West Asia'. Every level of this group is kept. |
| Order Region_West of USA  | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'West of USA '. Every level of this group is kept. |
| Order Region_Western Europe | one-hot (int64) | Integer 0/1 indicator for `Order Region` = 'Western Europe'. Every level of this group is kept. |
| Category Name_grouped | grouped text | Top 10 values of `Category Name` by row count, ties broken alphabetically. Other values mapped to `Other`. |
| Category Name_grouped_Camping & Hiking | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Camping & Hiking'. Every level of this group is kept. |
| Category Name_grouped_Cardio Equipment | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Cardio Equipment'. Every level of this group is kept. |
| Category Name_grouped_Cleats | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Cleats'. Every level of this group is kept. |
| Category Name_grouped_Electronics | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Electronics'. Every level of this group is kept. |
| Category Name_grouped_Fishing | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Fishing'. Every level of this group is kept. |
| Category Name_grouped_Indoor/Outdoor Games | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Indoor/Outdoor Games'. Every level of this group is kept. |
| Category Name_grouped_Men's Footwear | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = "Men's Footwear". Every level of this group is kept. |
| Category Name_grouped_Other | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Other'. Every level of this group is kept. |
| Category Name_grouped_Shop By Sport | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Shop By Sport'. Every level of this group is kept. |
| Category Name_grouped_Water Sports | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = 'Water Sports'. Every level of this group is kept. |
| Category Name_grouped_Women's Apparel | one-hot (int64) | Integer 0/1 indicator for `Category Name_grouped` = "Women's Apparel". Every level of this group is kept. |
| Order State_grouped | grouped text | Top 10 values of `Order State` by row count, ties broken alphabetically. Other values mapped to `Other`. |
| Order State_grouped_California | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'California'. Every level of this group is kept. |
| Order State_grouped_Distrito Federal | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Distrito Federal'. Every level of this group is kept. |
| Order State_grouped_Inglaterra | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Inglaterra'. Every level of this group is kept. |
| Order State_grouped_Isla de Francia | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Isla de Francia'. Every level of this group is kept. |
| Order State_grouped_Nueva Gales del Sur | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Nueva Gales del Sur'. Every level of this group is kept. |
| Order State_grouped_Nueva York | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Nueva York'. Every level of this group is kept. |
| Order State_grouped_Other | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Other'. Every level of this group is kept. |
| Order State_grouped_Renania del Norte-Westfalia | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Renania del Norte-Westfalia'. Every level of this group is kept. |
| Order State_grouped_San Salvador | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'San Salvador'. Every level of this group is kept. |
| Order State_grouped_Santo Domingo | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Santo Domingo'. Every level of this group is kept. |
| Order State_grouped_Texas | one-hot (int64) | Integer 0/1 indicator for `Order State_grouped` = 'Texas'. Every level of this group is kept. |
| Order Country_grouped | grouped text | Top 10 values of `Order Country` by row count, ties broken alphabetically. Other values mapped to `Other`. |
| Order Country_grouped_Alemania | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Alemania'. Every level of this group is kept. |
| Order Country_grouped_Australia | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Australia'. Every level of this group is kept. |
| Order Country_grouped_Brasil | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Brasil'. Every level of this group is kept. |
| Order Country_grouped_China | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'China'. Every level of this group is kept. |
| Order Country_grouped_Estados Unidos | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Estados Unidos'. Every level of this group is kept. |
| Order Country_grouped_Francia | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Francia'. Every level of this group is kept. |
| Order Country_grouped_India | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'India'. Every level of this group is kept. |
| Order Country_grouped_Italia | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Italia'. Every level of this group is kept. |
| Order Country_grouped_México | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'México'. Every level of this group is kept. |
| Order Country_grouped_Other | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Other'. Every level of this group is kept. |
| Order Country_grouped_Reino Unido | one-hot (int64) | Integer 0/1 indicator for `Order Country_grouped` = 'Reino Unido'. Every level of this group is kept. |
| late | target (int64) | Copied from `Late_delivery_risk`. The original column is also kept. |
| loss | target (int64) | 1 if `Benefit per order` < 0, else 0. Zero-profit orders are 0. |

## Outliers

| Column | Q1 | Q3 | IQR | Lower fence | Upper fence | Rows outside | Percent outside | Skewness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sales | 119.980003 | 299.950012 | 179.970009 | -149.975010 | 569.905025 | 488 | 0.2703% | 2.884249 |
| Benefit per order | 7.000000 | 64.800003 | 57.800003 | -79.700005 | 151.500008 | 18,942 | 10.4931% | -4.741834 |
| Order Item Quantity | 1.000000 | 3.000000 | 2.000000 | -2.000000 | 6.000000 | 0 | 0.0000% | 0.880252 |

`Benefit per order` negative outliers: 13,727.

Plot files:

- `A_preprocessing_eda/outputs/outlier_plots/box_Sales.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Sales_before.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Sales_after.png`
- `A_preprocessing_eda/outputs/outlier_plots/box_Benefit_per_order.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Benefit_per_order_before.png`
- `A_preprocessing_eda/outputs/outlier_plots/hist_Benefit_per_order_after.png`
- `A_preprocessing_eda/outputs/outlier_plots/box_Order_Item_Quantity.png`

## Encoding

| Column | Method | Levels | Notes |
| --- | --- | --- | --- |
| Shipping Mode | one-hot, all levels, integer 0/1 | 4 | All levels kept, in frequency order: 'Standard Class' (107,752), 'Second Class' (35,216), 'First Class' (27,814), 'Same Day' (9,737). |
| Market | one-hot, all levels, integer 0/1 | 5 | All levels kept, in frequency order: 'LATAM' (51,594), 'Europe' (50,252), 'Pacific Asia' (41,260), 'USCA' (25,799), 'Africa' (11,614). |
| Customer Segment | one-hot, all levels, integer 0/1 | 3 | All levels kept, in frequency order: 'Consumer' (93,504), 'Corporate' (54,789), 'Home Office' (32,226). |
| Type | one-hot, all levels, integer 0/1 | 4 | All levels kept, in frequency order: 'DEBIT' (69,295), 'TRANSFER' (49,883), 'PAYMENT' (41,725), 'CASH' (19,616). |
| Department Name | one-hot, all levels, integer 0/1 | 11 | All levels kept, in frequency order: 'Fan Shop' (66,861), 'Apparel' (48,998), 'Golf' (33,220), 'Footwear' (14,525), 'Outdoors' (9,686), 'Fitness' (2,479), 'Discs Shop' (2,026), 'Technology' (1,465), 'Pet Shop' (492), 'Book Shop' (405), 'Health and Beauty ' (362). |
| Order Region | one-hot, all levels, integer 0/1 | 23 | All levels kept, in frequency order: 'Central America' (28,341), 'Western Europe' (27,109), 'South America' (14,935), 'Oceania' (10,148), 'Northern Europe' (9,792), 'Southeast Asia' (9,539), 'Southern Europe' (9,431), 'Caribbean' (8,318), 'West of USA ' (7,993), 'South Asia' (7,731), 'Eastern Asia' (7,280), 'East of USA' (6,915), 'West Asia' (6,009), 'US Center ' (5,887), 'South of  USA ' (4,045), 'Eastern Europe' (3,920), 'West Africa' (3,696), 'North Africa' (3,232), 'East Africa' (1,852), 'Central Africa' (1,677), 'Southern Africa' (1,157), 'Canada' (959), 'Central Asia' (553). |
| Category Name | top-10 grouping, then one-hot of `Category Name_grouped`, integer 0/1 | 11 | 50 unique values; 10 kept; kept rows cover 88.8278%. Remaining rows mapped to `Other`: 20,168 (11.1722%). |
| Order State | top-10 grouping, then one-hot of `Order State_grouped`, integer 0/1 | 11 | 1,089 unique values; 10 kept; kept rows cover 19.3692%. Remaining rows mapped to `Other`: 145,554 (80.6308%). |
| Order Country | top-10 grouping, then one-hot of `Order Country_grouped`, integer 0/1 | 11 | 164 unique values; 10 kept; kept rows cover 55.4590%. Remaining rows mapped to `Other`: 80,405 (44.5410%). |

## Column dictionary

| Column | Dtype | Role | Downstream model | Note |
| --- | --- | --- | --- | --- |
| Type | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Days for shipment (scheduled) | int64 | numeric feature | late model, loss model, OLS | Known at order time. This is scheduled days, not the real shipping delay, so it is not leakage. |
| Benefit per order | float64 | target | OLS | Target for OLS, never a feature for loss or OLS. |
| Late_delivery_risk | int64 | target | late model | Late-delivery target. Identical to `late`. Not a feature. |
| Category Name | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Customer City | object | retained as text | none | Retained as text, not for modelling without a decision. |
| Customer Country | object | retained as text | none | Retained as text, not for modelling without a decision. |
| Customer Segment | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Customer State | object | retained as text | none | Retained as text, not for modelling without a decision. |
| Customer Zipcode | object | retained as text | none | Cast to text. 3 missing values filled with Unknown. Retained as text, not for modelling without a decision. |
| Department Name | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Latitude | float64 | numeric feature | late model, loss model, OLS | Geographic coordinate from the raw file. Unscaled. |
| Longitude | float64 | numeric feature | late model, loss model, OLS | Geographic coordinate from the raw file. Unscaled. |
| Market | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Order City | object | retained as text | none | Retained as text, not for modelling without a decision. |
| Order Country | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| order date (DateOrders) | object | derived date | none | Order timestamp, saved as ISO text `YYYY-MM-DD HH:MM:SS`. Use the integer date columns in models. Not derived from the shipping date. |
| Order Item Discount | float64 | numeric feature | late model, loss model, OLS | Unscaled discount amount from the raw file. |
| Order Item Discount Rate | float64 | numeric feature | late model, loss model, OLS | Unscaled discount rate from the raw file. Not standardised. |
| Order Item Product Price | float64 | numeric feature | late model, loss model, OLS | Unscaled. Kept because it matched `Product Price` on every row; that column was dropped. |
| Order Item Quantity | int64 | numeric feature | late model, loss model, OLS | Unscaled quantity. Absolute skewness 0.880252 did not exceed 1.0, so no log column was added. |
| Sales | float64 | numeric feature | late model, loss model, OLS | Unscaled item sales. `Sales_log` is the skew transform. This column is not an exact copy of `Order Item Total`. |
| Order Item Total | float64 | numeric feature | late model, loss model, OLS | Unscaled. Kept because it matched `Sales per customer` on every row; that column was dropped. |
| Order Region | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Order State | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| Order Status | object | retained as text | none | Retained as text, not for modelling without a decision. It may carry post-order information such as cancellation. |
| Product Name | object | retained as text | none | Retained as text, not for modelling without a decision. |
| Shipping Mode | object | categorical (text) | none | Original category text kept for Power BI and EDA. Models should use the one-hot columns. |
| order_year | int64 | derived date | late model, loss model, OLS | Calendar year of the order date. Not derived from the shipping date. |
| order_month | int64 | derived date | late model, loss model, OLS | Calendar month of the order date, 1 to 12. Not derived from the shipping date. |
| order_weekday | int64 | derived date | late model, loss model, OLS | Weekday of the order date, Monday = 0. Not derived from the shipping date. |
| order_weekday_name | object | BI only | none | English weekday name for Power BI. Models should use `order_weekday`. |
| Sales_log | float64 | transform | late model, loss model, OLS | `log1p` of `Sales`. The original `Sales` column is unchanged and unscaled. |
| Benefit_signed_log | float64 | transform | none | Signed log of `Benefit per order` for skew review. Not a feature for loss or OLS, because it is a transform of the profit target. |
| Category Name_grouped | object | categorical (text) | none | Grouped text of `Category Name`. Models should use the one-hot columns, not this text. |
| Order State_grouped | object | categorical (text) | none | Grouped text of `Order State`. Models should use the one-hot columns, not this text. |
| Order Country_grouped | object | categorical (text) | none | Grouped text of `Order Country`. Models should use the one-hot columns, not this text. |
| Shipping Mode_First Class | int64 | one-hot | late model, loss model, OLS | One-hot level of `Shipping Mode`. Drop one level from this group before a model with an intercept. |
| Shipping Mode_Same Day | int64 | one-hot | late model, loss model, OLS | One-hot level of `Shipping Mode`. Drop one level from this group before a model with an intercept. |
| Shipping Mode_Second Class | int64 | one-hot | late model, loss model, OLS | One-hot level of `Shipping Mode`. Drop one level from this group before a model with an intercept. |
| Shipping Mode_Standard Class | int64 | one-hot | late model, loss model, OLS | One-hot level of `Shipping Mode`. Drop one level from this group before a model with an intercept. |
| Market_Africa | int64 | one-hot | late model, loss model, OLS | One-hot level of `Market`. Drop one level from this group before a model with an intercept. |
| Market_Europe | int64 | one-hot | late model, loss model, OLS | One-hot level of `Market`. Drop one level from this group before a model with an intercept. |
| Market_LATAM | int64 | one-hot | late model, loss model, OLS | One-hot level of `Market`. Drop one level from this group before a model with an intercept. |
| Market_Pacific Asia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Market`. Drop one level from this group before a model with an intercept. |
| Market_USCA | int64 | one-hot | late model, loss model, OLS | One-hot level of `Market`. Drop one level from this group before a model with an intercept. |
| Customer Segment_Consumer | int64 | one-hot | late model, loss model, OLS | One-hot level of `Customer Segment`. Drop one level from this group before a model with an intercept. |
| Customer Segment_Corporate | int64 | one-hot | late model, loss model, OLS | One-hot level of `Customer Segment`. Drop one level from this group before a model with an intercept. |
| Customer Segment_Home Office | int64 | one-hot | late model, loss model, OLS | One-hot level of `Customer Segment`. Drop one level from this group before a model with an intercept. |
| Type_CASH | int64 | one-hot | late model, loss model, OLS | One-hot level of `Type`. Drop one level from this group before a model with an intercept. |
| Type_DEBIT | int64 | one-hot | late model, loss model, OLS | One-hot level of `Type`. Drop one level from this group before a model with an intercept. |
| Type_PAYMENT | int64 | one-hot | late model, loss model, OLS | One-hot level of `Type`. Drop one level from this group before a model with an intercept. |
| Type_TRANSFER | int64 | one-hot | late model, loss model, OLS | One-hot level of `Type`. Drop one level from this group before a model with an intercept. |
| Department Name_Apparel | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Book Shop | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Discs Shop | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Fan Shop | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Fitness | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Footwear | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Golf | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Health and Beauty  | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Outdoors | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Pet Shop | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Department Name_Technology | int64 | one-hot | late model, loss model, OLS | One-hot level of `Department Name`. Drop one level from this group before a model with an intercept. |
| Order Region_Canada | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Caribbean | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Central Africa | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Central America | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Central Asia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_East Africa | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_East of USA | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Eastern Asia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Eastern Europe | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_North Africa | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Northern Europe | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Oceania | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_South America | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_South Asia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_South of  USA  | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Southeast Asia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Southern Africa | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Southern Europe | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_US Center  | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_West Africa | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_West Asia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_West of USA  | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Order Region_Western Europe | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Region`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Camping & Hiking | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Cardio Equipment | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Cleats | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Electronics | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Fishing | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Indoor/Outdoor Games | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Men's Footwear | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Other | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Shop By Sport | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Water Sports | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Category Name_grouped_Women's Apparel | int64 | one-hot | late model, loss model, OLS | One-hot level of `Category Name_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_California | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Distrito Federal | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Inglaterra | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Isla de Francia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Nueva Gales del Sur | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Nueva York | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Other | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Renania del Norte-Westfalia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_San Salvador | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Santo Domingo | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order State_grouped_Texas | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order State_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Alemania | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Australia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Brasil | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_China | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Estados Unidos | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Francia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_India | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Italia | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_México | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Other | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| Order Country_grouped_Reino Unido | int64 | one-hot | late model, loss model, OLS | One-hot level of `Order Country_grouped`. Drop one level from this group before a model with an intercept. |
| late | int64 | target | late model | Late-delivery target. Identical to `Late_delivery_risk`. Not a feature. |
| loss | int64 | target | loss model | Loss target: 1 if `Benefit per order` < 0, else 0. Not a feature. |

## Leakage summary

| Column | Leaks for | Why | In the cleaned file |
| --- | --- | --- | --- |
| Delivery Status | late model | Known only after the delivery outcome is recorded, and it restates whether the order was late. | No. Dropped. |
| Days for shipping (real) | late model | The actual days in transit are known only after delivery, and lateness is that value compared with the scheduled days. | No. Dropped. |
| shipping date (DateOrders) | late model | The ship timestamp is known only after dispatch, so it is not available at order time and leaks the late-delivery target. | No. Dropped. |
| Order Profit Per Order | loss model, OLS | It defines profit directly. Benefit per order is kept because it is the OLS target and the source of the loss target. | No. Dropped. |
| Order Item Profit Ratio | loss model, OLS | It expresses profit relative to the item value, so it defines the loss and profit targets. | No. Dropped. |
| Benefit per order | loss model, OLS (if used as a feature) | It is the profit target and the source of `loss`. Using it as a feature would leak both targets. | Yes. Kept as the OLS target. Not a feature for loss or OLS. |

`Days for shipment (scheduled)` is not in this table because it is known at order time and remains available to every model.

## Verification results

Output of `A_preprocessing_eda/phase1_validate.py` (exit code 0):

```
PASS row count equals raw row count
PASS zero nulls in the cleaned file
PASS dropped column absent: Product Description
PASS dropped column absent: Order Zipcode
PASS dropped column absent: Product Status
PASS dropped column absent: Product Image
PASS dropped column absent: Customer Password
PASS dropped column absent: Customer Email
PASS dropped column absent: Customer Fname
PASS dropped column absent: Customer Lname
PASS dropped column absent: Customer Street
PASS dropped column absent: Customer Id
PASS dropped column absent: Order Customer Id
PASS dropped column absent: Order Id
PASS dropped column absent: Order Item Id
PASS dropped column absent: Category Id
PASS dropped column absent: Department Id
PASS dropped column absent: Product Card Id
PASS dropped column absent: Product Category Id
PASS dropped column absent: Order Item Cardprod Id
PASS dropped column absent: Delivery Status
PASS dropped column absent: Days for shipping (real)
PASS dropped column absent: shipping date (DateOrders)
PASS dropped column absent: Order Profit Per Order
PASS dropped column absent: Order Item Profit Ratio
PASS dropped column absent: Sales per customer
PASS dropped column absent: Product Price
PASS no column name contains 'shipping date'
PASS required columns present
PASS Days for shipment (scheduled) is present
PASS late is an integer 0/1 column with no nulls
PASS loss is an integer 0/1 column with no nulls
PASS loss equals Benefit per order < 0 on every row
PASS late equals Late_delivery_risk on every row
PASS Benefit per order equals the raw values
PASS Sales equals the raw values
PASS Order Item Quantity equals the raw values
PASS negative Benefit per order count equals the raw count
PASS one-hot group Category Name_grouped: integer 0/1 and row sums equal 1
PASS one-hot group Order State_grouped: integer 0/1 and row sums equal 1
PASS one-hot group Order Country_grouped: integer 0/1 and row sums equal 1
PASS one-hot group Customer Segment: integer 0/1 and row sums equal 1
PASS one-hot group Department Name: integer 0/1 and row sums equal 1
PASS one-hot group Shipping Mode: integer 0/1 and row sums equal 1
PASS one-hot group Order Region: integer 0/1 and row sums equal 1
PASS one-hot group Market: integer 0/1 and row sums equal 1
PASS one-hot group Type: integer 0/1 and row sums equal 1
PASS order_year matches the parsed order date
PASS order_month matches the parsed order date
PASS order_weekday matches the parsed order date (Monday = 0)
PASS order_weekday_name matches Monday = 0
PASS Sales_log equals log1p(Sales)
PASS Benefit_signed_log equals sign(x) * log1p(abs(x))
PASS no duplicate column names
PASS duplicate rows stated: raw=0, cleaned=0. Row count equals the raw file, so no rows were deleted.
PASS measurement columns are not standardised (mean near 0 and std near 1)
SUMMARY: 56 passed, 0 failed
```

## Discrepancies

- The project brief expected order dates from 1 Jan 2015 to 9 Sep 2017. The minimum is 2015-01-01 00:00:00. The maximum is 2018-01-31 23:38:00, which is after 9 Sep 2017. No rows were removed to force the expected window.

## Assumptions and open decisions

- `random_state = 42` is set, and no step samples or shuffles rows. The pipeline is deterministic.
- No rows are deleted, including negative-profit rows. Nothing is scaled or split in this phase.
- Original category text is kept beside the one-hot columns so Power BI can read the labels. Models should use the 0/1 columns. No reference level is dropped here.
- `TOP_N = 10`. Ties in frequency are broken by sorting the label alphabetically. Coverage of the kept labels is in the encoding table. `Order State` coverage is low, so most state rows fall in `Other`; the constant was not changed.
- Near-duplicates: `Sales per customer` was dropped in favour of `Order Item Total` because the two matched within tolerance. `Product Price` was dropped in favour of `Order Item Product Price` by the same exact-match rule, keeping the earlier protected column.
- `Order Status` is retained as text and is not encoded. Levels such as cancellation, suspected fraud, and payment review can be known only after the order is placed. Whether any model may use it is an open decision. It was not dropped.
- A log column for `Order Item Quantity` is added only if absolute skewness exceeds 1.0. The decision and both skewness figures are in Step 5.
- One-hot columns inside each group are sorted alphabetically. This does not drop a level.
- Weekday names come from a fixed English list with Monday = 0, so they do not depend on the machine locale.
- `Customer Zipcode` nulls (3) were filled with `Unknown` and the rest were written as text. The column was not one-hot encoded.
- Category labels were not stripped. Source values with leading or trailing spaces were kept so the original text stays faithful to the file.
- Labels whose stored text is not equal to their stripped text: `Department Name`: 'Health and Beauty '; `Order Region`: 'South of  USA ', 'US Center ', 'West of USA '; `Category Name`: 'Baby ', 'Books ', 'CDs ', 'Cameras '; `Product Name`: 'DVDs ', 'Smart watch ', 'Sports Books ', 'Toys '
- `Customer Country` and the other text-only columns were not encoded because they are outside the specified one-hot and top-N lists, even where cardinality is low.
- The order date was parsed as month/day/year hour:minute because that format accepted every value. A day-first format does not.
- Cleaned duplicate rows are 0, the same as the raw file (0). No row was deleted. Dropping the identifier columns did not create duplicate rows in this file.

## Limitations

The data come from a single company (DataCo). Orders in this file run from 2015-01-01 00:00:00 to 2018-01-31 23:38:00. Loss is defined only as `Benefit per order` < 0, so an order with profit of exactly zero is not a loss. Columns that leak the late-delivery outcome, and columns that define profit other than the target itself, were removed. `Benefit per order` remains because it is the OLS target. The cleaned file is not scaled and is not split into train and test.
