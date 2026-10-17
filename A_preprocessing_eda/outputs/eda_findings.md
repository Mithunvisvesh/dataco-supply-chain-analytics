# Phase 2a: EDA findings

- Run date: 2026-10-17
- Script: `A_preprocessing_eda/phase2a_eda.py`
- Input: `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`
- Rows used: 180,519
- Overall late rate: 54.83%
- Overall loss rate: 18.71%

## Summary statistics

Five-number summary plus mean, standard deviation, skewness and kurtosis for the focus numeric columns.

| Column | Mean | Std | Min | Q1 | Median | Q3 | Max | Skew | Kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sales | 203.7721 | 132.2731 | 9.9900 | 119.9800 | 199.9200 | 299.9500 | 1,999.9900 | 2.8842 | 23.9366 |
| Benefit per order | 21.9750 | 104.4335 | -4,274.9800 | 7.0000 | 31.5200 | 64.8000 | 911.8000 | -4.7418 | 71.3773 |
| Order Item Discount | 20.6647 | 21.8009 | 0.0000 | 5.4000 | 14.0000 | 29.9900 | 500.0000 | 3.0398 | 25.2313 |
| Order Item Discount Rate | 0.1017 | 0.0704 | 0.0000 | 0.0400 | 0.1000 | 0.1600 | 0.2500 | 0.3409 | -0.9012 |
| Days for shipment (scheduled) | 2.9318 | 1.3744 | 0.0000 | 2.0000 | 4.0000 | 4.0000 | 4.0000 | -0.7320 | -1.0229 |
| Order Item Quantity | 2.1276 | 1.4535 | 1.0000 | 1.0000 | 1.0000 | 3.0000 | 5.0000 | 0.8803 | -0.7537 |

Discount column used below is `Order Item Discount Rate` (rate) and `Order Item Discount` (amount). Shipping days means `Days for shipment (scheduled)`.

Mean scheduled days: 2.9318. Mean discount rate: 0.1017.

## Findings (plain language)

### F1. Shipping mode dominates late delivery.

`First Class` has the highest late rate (95.32%), while `Standard Class` is lowest (38.07%). This is the clearest categorical signal for the late-delivery question.

Plot: `bar_late_rate_by_Shipping_Mode.png`
### F2. Late rate also differs by market and region.

Markets range from `LATAM` (54.36%) to `Europe` (55.21%). Among regions, `Central Africa` is highest (57.96%) and `Canada` is lowest (48.80%).

Plot: `bar_late_rate_by_Market.png / bar_late_rate_by_Order_Region.png`
### F3. Loss rate is fairly similar across customer segments.

Loss shares: Home Office 18.85%, Corporate 18.82%, Consumer 18.60%. Segment may still be worth testing, but the effect looks smaller than shipping mode for lateness.

Plot: `bar_loss_rate_by_Customer_Segment.png`
### F4. Profit is heavy-tailed and often negative.

`Benefit per order` mean 21.9750, median 31.5200, skewness -4.7418. Box/violin plots by market and shipping mode show wide spreads; OLS should expect assumption issues.

Plot: `box_profit_by_Market.png / hist_Benefit_per_order.png`
### F5. Discount rate vs loss looks weak on averages; watch the scatter with profit.

The discount-vs-profit scatter still shows many deep losses at higher discount rates, but the mean discount rate gap is small: loss 0.1022 vs non-loss 0.1015. B should report effect size for T6, not only the p-value.

Plot: `scatter_discount_vs_profit.png / overlay_discount_loss.png`
### F6. Scheduled shipping days differ for late vs on-time orders.

Mean scheduled days: late 2.4711, on-time 3.4911. This is a candidate feature for D (known at order time, not leakage).

Plot: `overlay_scheduled_days_late.png`
### F7. Sales are right-skewed; the log transform helps for visuals.

`Sales` skewness 2.8842. Use `Sales_log` for distribution plots; keep raw `Sales` for modelling unless C chooses otherwise.

Plot: `hist_compare_Sales.png`
### F8. Order volume and late rate move over the calendar.

Busiest month in this extract: 2016-10 (5,398 orders). Highest monthly late rate: 2016-06 (56.85%). B can use month as a slicer in Power BI; D may try `order_month` / `order_year` as features.

Plot: `time_orders_and_late_rate.png`


## Correlation notes (for C)

Strongest absolute Pearson correlations with `late` (excluding itself):
- `Days for shipment (scheduled)`: r = -0.3694
- `Order Item Total`: r = -0.0038
- `Benefit per order`: r = -0.0037
- `Sales`: r = -0.0036
- `Order Item Product Price`: r = -0.0022

Strongest absolute Pearson correlations with `loss` (excluding itself):
- `Benefit per order`: r = -0.6291
- `Days for shipment (scheduled)`: r = 0.0040
- `Order Item Discount Rate`: r = 0.0038
- `Longitude`: r = 0.0033
- `Order Item Total`: r = -0.0019

Remember: `Sales` and `Order Item Total` are highly collinear (kept both in Phase 1). Check VIF before OLS.

## Segments worth testing (for B)

These match the planned T1–T6 tests and are backed by the plots above.

1. Shipping Mode vs late — spread 38.07% to 95.32% (`Standard Class` vs `First Class`). Plot: `bar_late_rate_by_Shipping_Mode.png`.
2. Market vs late — spread 54.36% to 55.21% (`LATAM` vs `Europe`). Plot: `bar_late_rate_by_Market.png`.
3. Customer Segment vs loss — rates `Home Office` 18.85%, `Corporate` 18.82%, `Consumer` 18.60%. Plot: `bar_loss_rate_by_Customer_Segment.png`.
4. Profit across Markets (ANOVA / Kruskal–Wallis) — see `box_profit_by_Market.png` / `violin_profit_by_Market.png`.
5. Profit: late vs on-time (Welch t-test) — mean profit late 21.6217, on-time 22.4038.
6. Discount rate: loss vs profit (Welch t-test) — mean discount loss 0.1022, profit 0.1015. Plot: `overlay_discount_loss.png`.

## Plot index

- `A_preprocessing_eda/outputs/eda_plots/hist_Sales.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_Benefit_per_order.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_Order_Item_Discount.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_Order_Item_Discount_Rate.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_Days_for_shipment_scheduled.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_Order_Item_Quantity.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_compare_Sales.png`
- `A_preprocessing_eda/outputs/eda_plots/hist_compare_Benefit_per_order.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_Shipping_Mode.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_Market.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_Customer_Segment.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_Type.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_Department_Name.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_Order_Region.png`
- `A_preprocessing_eda/outputs/eda_plots/heatmap_correlation.png`
- `A_preprocessing_eda/outputs/eda_plots/scatter_discount_vs_profit.png`
- `A_preprocessing_eda/outputs/eda_plots/box_profit_by_Shipping_Mode.png`
- `A_preprocessing_eda/outputs/eda_plots/violin_profit_by_Shipping_Mode.png`
- `A_preprocessing_eda/outputs/eda_plots/box_profit_by_Market.png`
- `A_preprocessing_eda/outputs/eda_plots/violin_profit_by_Market.png`
- `A_preprocessing_eda/outputs/eda_plots/box_profit_by_Customer_Segment.png`
- `A_preprocessing_eda/outputs/eda_plots/violin_profit_by_Customer_Segment.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_late_rate_by_Shipping_Mode.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_late_rate_by_Market.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_late_rate_by_Order_Region.png`
- `A_preprocessing_eda/outputs/eda_plots/bar_loss_rate_by_Customer_Segment.png`
- `A_preprocessing_eda/outputs/eda_plots/overlay_scheduled_days_late.png`
- `A_preprocessing_eda/outputs/eda_plots/overlay_discount_late.png`
- `A_preprocessing_eda/outputs/eda_plots/overlay_discount_loss.png`
- `A_preprocessing_eda/outputs/eda_plots/time_orders_and_late_rate.png`

## Limitations

EDA is descriptive only. With ~180k rows, tiny differences can look “significant” in later tests; prefer effect sizes. Leakage columns stay excluded. No modelling was done in Phase 2a.
