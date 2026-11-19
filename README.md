# Analytics-Report-Insights-Documentation
A Python tool that turns sales data into a stakeholder-ready HTML analytics report with KPIs, charts, insights, and prioritized recommendations, all auto-generated from the data, with cleaning and methodology.
Analytics Report & Insights Documentation

A Python tool that turns raw sales data into a professional, stakeholder-ready analytics report: executive summary, KPIs, visualized findings, insights, and prioritized, actionable recommendations. The report is a single self-contained HTML file that can be opened in any browser or printed to PDF.

Objective

Create comprehensive analytics reports that include an executive summary, findings, visualizations, insights and actionable recommendations for business stakeholders.

What the Report Contains
Section	Content
1. Executive Summary	Six KPI cards with year-on-year change, headline bullets and the top three recommended actions
2. Key Findings and Insights	Seven findings, each with a chart, a plain-language Finding and a Why it matters statement
3. Recommendations	Prioritized action table (High, Medium, Low) with rationale, estimated impact and the KPI to track
4. Methodology and Data Quality	Metric definitions and a log of every cleaning step applied
5. Limitations	Caveats stakeholders should know before acting
Analyses covered
Revenue trend and seasonality (monthly, by year)
Category revenue and profit margin
Regional performance
Channel mix shift and margin by channel
Discount impact on margin and loss-making orders, with a discount-cap scenario
Customer concentration, repeat purchasing and return rates
Product Pareto analysis and loss-making products

All text in the findings and recommendations is generated from the data, so it updates automatically when you supply a different dataset.

Project Structure
analytics_report_project/
├── analytics_report.py   # main script
├── requirements.txt      # dependencies
├── README.md             # this file
└── output/               # created when you run the script
    ├── analytics_report.html   # the report
    ├── insights.json           # KPIs, insights and recommendations (machine-readable)
    ├── cleaned_data.csv        # cleaned dataset used in the analysis
    └── figures/                # all 7 charts as PNG files
Setup

Python 3.9 or later is recommended.

bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
How to Run
bash
python analytics_report.py

Then open output/analytics_report.html in your browser. To make a PDF, use the browser's Print option and choose "Save as PDF" (the page has print-friendly styling).

Options
Option	Description	Default
--csv	Path to your own sales CSV	synthetic demo data
--output	Output folder	output
--company	Company name shown in the report header	Northwind Retail
--seed	Random seed for the demo data	42
--discount-cap	Discount cap used in the recommendation scenario	0.20

Example:

bash
python analytics_report.py --csv my_sales.csv --company "Acme Ltd" --discount-cap 0.15
Data

By default the script generates a synthetic, reproducible dataset of about 12,000 retail orders (January 2024 to December 2025) with built-in seasonality, growth, regional differences, a channel shift, discounting, returns, and deliberately injected quality problems (duplicates and missing values) so the cleaning step has real work to do. Because it is synthetic, the specific numbers are illustrative, not real business results.

Using your own data

Provide a CSV with these columns (names are not case-sensitive; spaces are converted to underscores):

Column	Meaning
order_id	Unique order identifier
order_date	Date of the order
customer_id	Customer identifier
region, category, product, segment, channel	Descriptive attributes
units	Units sold
unit_price	List price per unit, before discount
unit_cost	Cost per unit
discount	Discount as a fraction (0.10) or a percentage (10)
returned	1/0, true/false or yes/no
Data Preparation Steps
Validates required columns and parses dates.
Removes duplicate orders and rows with invalid units, prices or costs.
Fills missing discounts with the category median and other missing labels with "Unknown" (excluded when ranking best and worst performers).
Converts percentage discounts to fractions when needed.
Flags statistical outliers (kept, because large orders are valid).
Calculates revenue, cost and profit, and excludes returned orders from net figures.
Metric Definitions
Net revenue = units x list price x (1 - discount), excluding returned orders.
Net profit = net revenue minus product cost, excluding returned orders.
Margin = net profit / net revenue.
Year-on-year compares the latest calendar year in the data with the previous one.
Sample Output (demo data, seed 42)

Headlines from the executive summary:

2025 net revenue was $2.98M (+9.5% vs 2024) and net profit $611K (+10.9%), at a 20.5% margin.
Technology is the largest category (65% of revenue) but has the thinnest margin (18.0%).
Margin falls from 29.2% on undiscounted orders to -1.7% on orders discounted more than 20%, and about $56K of discount spend sits above a 20% cap.
Online grew from 30% to 38% of revenue; Retail has the best channel margin (22.1%) and Wholesale the lowest (17.4%).
The top 10% of customers drive 61% of revenue, and 45% of customers ordered only once.

Top recommendations: cap discounts at 20%, protect margin in Technology, steer growth toward higher-margin channels, improve South region margin, prepare for the Q4 peak, reduce Furniture returns, and launch a retention programme.

Good Reporting Practices Demonstrated
Lead with the answer: the executive summary comes first, in business language.
Finding, then implication: every chart pairs "what we found" with "why it matters".
Actionable recommendations: each has a priority, rationale, estimated impact and a KPI to track.
Transparency: definitions, data-cleaning log and limitations are documented.
Honest estimates: impact figures are labelled as upper bounds or illustrations, and cover the whole analysed period.
Limitations
Findings show association, not proven cause.
Impact estimates ignore changes in customer behaviour.
Returned orders are assumed to lose all revenue and profit.
Demo data is synthetic and for demonstration only.
Possible Extensions
Add a PDF export (for example with WeasyPrint).
Add forecasting for the next quarter.
Add customer segmentation (RFM) and churn analysis.
Build an interactive dashboard with Streamlit or Plotly.
Schedule the report to regenerate and email automatically.
Tools Used

Python, pandas, NumPy, Matplotlib, seaborn.
