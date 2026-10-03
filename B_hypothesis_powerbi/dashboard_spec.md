# Power BI dashboard spec (Person B)

Load `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`. Do not edit that file.
Set types, then add measures. More visuals than slicers. One theme, a border on each block,
and a one-line insight next to key charts.

## Measures

| Measure | Rule |
| --- | --- |
| Total orders | Count of rows (order lines) |
| Late % | Average of `late` |
| Loss % | Average of `loss` |
| Total sales | Sum of `Sales` |
| Total profit | Sum of `Benefit per order` |

Use the **text** columns for slicers and axes (`Shipping Mode`, `Market`, `Customer Segment`,
`order_year`, `Order Region`). Do not drag one-hot dummy columns onto the canvas.

## Pages

### Page 1 — Overview (due Sunday night)

- KPI cards: total orders, late %, loss %, total sales, total profit
- Orders by month (`order_year` + `order_month`)
- Map or bar by `Market`
- Slicers: Market, year

### Page 2 — Delivery performance (due Sunday night)

- Late rate by `Shipping Mode` (this is the only large statistical effect; lead with it)
- Late rate by `Order Region` and by month
- Scheduled shipping days (`Days for shipment (scheduled)`)
- Slicer: shipping mode

### Page 3 — Profit and loss (due Sunday night)

- Loss % by `Customer Segment` and `Category Name` (or `Category Name_grouped`)
- Profit by `Market`
- Discount vs profit scatter (`Order Item Discount Rate` vs `Benefit per order`; sample if needed)
- Slicer: segment

### Page 4 — Model results (Monday 5 Oct, 11 am)

Wait for C and D to push:

- Model comparison table
- Odds-ratio / coefficient charts
- OLS actual vs predicted

## Numbers the tests support (do not over-claim)

- Shipping mode is associated with lateness (Cramer's V ≈ 0.46). First Class ~95% late; Standard Class ~38%.
- Market vs late, segment vs loss, late vs profit, and discount vs loss **do not reject H0** at 0.05.
  Do not write dashboard titles that claim they "drive" anything.
