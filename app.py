from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load('pipeline.pkl')

cols = ['WorkingCapital/Equity',
 'PersistentEPSintheLastFourSeasons',
 'BorrowingDependency',
 'NetValueGrowthRate',
 'InterestBearingDebtInterestRate',
 'ROA(C)BeforeInterestAndDepreciationBeforeInterest',
 'Cash/TotalAssets',
 'NonIndustryIncomeAndExpenditure/Revenue',
 'NetValuePerShare(B)',
 'TotalDebt/TotalNetWorth']

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        working_capital_equity = request.form['working_capital_equity']
        persistent_eps_in_last_4_seasons = request.form['persistent_eps_in_last_4_seasons']
        borrowing_dependency = request.form['borrowing_dependency']
        net_value_growth_rate = request.form['net_value_growth_rate']
        interest_bearing_debt_interest_rate = request.form['interest_bearing_debt_interest_rate']
        roa_c_before_interest_and_depreciation_before_interest = request.form['roa_c_before_interest_and_depreciation_before_interest']
        cash_total_assets = request.form['cash_total_assets']
        non_industry_income_and_expenditure_revenue = request.form['non_industry_income_and_expenditure_revenue']
        net_value_per_share_b = request.form['net_value_per_share_b']
        total_debt_total_net_worth = request.form['total_debt_total_net_worth']
        data = pd.DataFrame([[working_capital_equity,
                 persistent_eps_in_last_4_seasons,
                 borrowing_dependency,
                 net_value_growth_rate,
                 interest_bearing_debt_interest_rate,
                 roa_c_before_interest_and_depreciation_before_interest,
                 cash_total_assets,
                 non_industry_income_and_expenditure_revenue,
                 net_value_per_share_b,
                 total_debt_total_net_worth]],columns=cols)
        pred = pipeline.predict(data)
        if pred == 0:
            return render_template('index.html',prediction_text="The company with the specified details will not become bankrupt.")
        elif pred == 1:
            return render_template('index.html',prediction_text="The company with the specified details will become bankrupt in the near future.")
        
if __name__ == '__main__':
    app.run(port=8000)