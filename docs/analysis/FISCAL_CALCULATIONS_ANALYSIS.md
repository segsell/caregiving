# Fiscal Calculations for Caregiving Model: Current Capabilities and Extensions

## Executive Summary

This document systematically documents:
1. **Current Ingredients**: What fiscal data is available from `budget_constraint` and simulated data
2. **Target Quantities**: What fiscal impacts can be measured
3. **Computation Methods**: How to calculate these quantities
4. **Missing Components**: What additional data/functionality would enable more comprehensive fiscal analysis

---

## 1. Current Ingredients: Available Data

### 1.1 Data Returned from `budget_constraint` (in `aux` dictionary)

The `budget_constraint` function returns an `aux` dictionary containing the following fiscal variables (all normalized by `wealth_unit`):

#### Revenue Components
- **`income_tax`**: Total household income tax (includes Soli and church tax)
- **`income_tax_single`**: Income tax if filing as single (for decomposition)
- **`own_ssc`**: Own social security contributions (pension/unemployment + health/LTC)
- **`partner_ssc`**: Partner social security contributions
- **`total_tax_revenue`**: Total revenue = `income_tax` + `own_ssc` + `partner_ssc`

#### Expenditure Components
- **`child_benefits`**: Annual child benefits paid by government
- **`care_benefits_and_costs`**: Net care benefits (positive) minus formal care costs (negative)
  - **Note**: Currently aggregated; positive values are informal care cash benefits, negative values are formal care costs
- **`household_unemployment_benefits`**: Unemployment benefits (means-tested)

#### Aggregate Budget
- **`government_expenditures`**: Total expenditures = `child_benefits` + `max(care_benefits_and_costs, 0)` + `household_unemployment_benefits`
- **`net_government_budget`**: Net budget = `total_tax_revenue` - `government_expenditures`

#### Supporting Income Variables
- **`gross_labor_income`**: Own gross labor income
- **`gross_retirement_income`**: Own gross pension income
- **`gross_partner_income`**: Partner gross labor income
- **`gross_partner_pension`**: Partner gross pension income
- **`own_income_after_ssc`**: Own income after SSC (before income tax)
- **`net_hh_income`**: Total household net income (after all taxes, including interest)
- **`hh_net_income_wo_interest`**: Household net income excluding interest

### 1.2 Data Available in Simulated DataFrame

From `simulate_scenario()`:
- **State variables**: `period`, `age`, `education`, `health`, `partner_state`, `care_demand`, `choice`, `experience`
- **Wealth variables**: `assets_begin_of_period`, `savings`
- **Income variables**: All variables from `aux` dictionary are available as columns
- **Derived variables**: `working_hours`, `exp_years`, `is_retired`, `actual_retirement_age`
- **Consumption**: `consumption`, `utility`

### 1.3 SSC Component Breakdown (from `tax_and_ssc.py`)

**Pension/Unemployment SSC** (`calc_pension_unempl_contr`):
- Rate: 10.5% (9.3% pension + 1.2% unemployment)
- Threshold: €6,823.50/month × 12 = €81,882/year
- Applied to: Workers only (not retirees)

**Health/LTC SSC** (`calc_health_ltc_contr`):
- Rate: 9.625% (7% health + 1.1% additional health + 1.525% LTC)
- Threshold: €4,687.50/month × 12 = €56,250/year
- Applied to: Both workers and retirees

**Income Tax** (`calc_inc_tax_for_single_income`):
- Progressive brackets with quadratic/linear rates
- Includes Soli (5% of income tax) and church tax (9% × 50% population share = 4.5% effective)

### 1.4 Care Benefits and Costs Structure

**Informal Care Cash Benefits** (positive, government expenditure):
- `informal_care_cash_benefits_light`: Monthly benefit for light informal care
- `informal_care_cash_benefits_intensive`: Monthly benefit for intensive informal care
- Annualized: `monthly_benefit × 12 × care_type_indicator`

**Formal Care Costs** (negative, government expenditure):
- `formal_care_costs[period]`: Period-specific monthly formal care cost
- Annualized: `formal_care_costs[period] × 12 × formal_care_indicator`
- **Note**: Currently, formal care costs are treated as government expenditures (subsidized care)

---

## 2. Target Quantities: What Can Be Measured

### 2.1 Aggregate Fiscal Impacts

#### Lifetime Fiscal Impact per Agent
- **Net present value (NPV) of government budget** over agent's lifetime
- **Average annual fiscal impact** by age/period
- **Fiscal impact by education/health/partner status**

#### Counterfactual Comparisons
- **Baseline vs. counterfactual**: Difference in fiscal impacts
- **Caregiving leave policies**: Additional costs of top-up payments
- **Formal vs. informal care**: Fiscal implications of care arrangement choices

### 2.2 Revenue Decomposition

#### By Source
- **Income tax revenue**: Total, by tax bracket, by household type
- **SSC revenue**: Own vs. partner, pension/unemployment vs. health/LTC
- **Revenue by employment status**: Working vs. retired vs. unemployed

#### By Demographic Group
- **By education**: High vs. low education
- **By health status**: Good vs. bad health
- **By partner status**: Single vs. partnered
- **By age/period**: Lifecycle patterns

### 2.3 Expenditure Decomposition

#### By Type
- **Child benefits**: Total, per child, by household type
- **Care benefits**: Informal care cash benefits (light vs. intensive)
- **Formal care costs**: Government subsidies for formal care
- **Unemployment benefits**: Total, by household type, means-tested amounts

#### By Demographic Group
- **By care arrangement**: No care vs. informal vs. formal
- **By employment status**: Impact of labor supply decisions
- **By age/period**: Lifecycle patterns

### 2.4 Behavioral Responses and Fiscal Spillovers

#### Labor Supply Effects
- **Tax revenue changes** due to reduced labor supply (caregiving)
- **SSC revenue changes** due to reduced working hours
- **Unemployment benefit changes** due to exit from labor force

#### Care Arrangement Effects
- **Substitution effects**: Informal → formal care (cost implications)
- **Care benefit changes**: Light → intensive informal care

#### Retirement Effects
- **Pension expenditure**: Changes in retirement timing
- **SSC revenue**: Loss of contributions due to early retirement

---

## 3. Computation Methods

### 3.1 Aggregate Fiscal Calculations

#### Method 1: Direct Aggregation from Simulated Data

```python
# Aggregate net government budget by age
fiscal_by_age = sim_df.groupby('age').agg({
    'net_government_budget': 'mean',
    'total_tax_revenue': 'mean',
    'government_expenditures': 'mean',
})

# Lifetime NPV (assuming discount rate r)
discount_factor = (1 + r) ** (-sim_df['period'])
lifetime_npv = (sim_df['net_government_budget'] * discount_factor).groupby('agent').sum()
```

**Pros**: Simple, uses existing data
**Cons**: Requires discount rate assumption, no decomposition

#### Method 2: Decomposed Aggregation

```python
# Revenue decomposition
revenue_by_source = sim_df.groupby('age').agg({
    'income_tax': 'mean',
    'own_ssc': 'mean',
    'partner_ssc': 'mean',
})

# Expenditure decomposition
expenditure_by_type = sim_df.groupby('age').agg({
    'child_benefits': 'mean',
    'care_benefits_and_costs': lambda x: x[x > 0].mean(),  # Only benefits
    'household_unemployment_benefits': 'mean',
})
```

**Pros**: Detailed breakdown, identifies key drivers
**Cons**: Need to handle `care_benefits_and_costs` sign correctly

#### Method 3: Counterfactual Comparison

```python
# Compare baseline vs. counterfactual
baseline_fiscal = baseline_df.groupby('age')['net_government_budget'].mean()
counterfactual_fiscal = cf_df.groupby('age')['net_government_budget'].mean()
fiscal_impact = counterfactual_fiscal - baseline_fiscal

# Decompose impact
revenue_impact = (cf_df['total_tax_revenue'] - baseline_df['total_tax_revenue']).groupby('age').mean()
expenditure_impact = (cf_df['government_expenditures'] - baseline_df['government_expenditures']).groupby('age').mean()
```

**Pros**: Policy-relevant, shows net effects
**Cons**: Requires running both scenarios

### 3.2 Care-Specific Fiscal Calculations

#### Method 1: Care Arrangement Decomposition

```python
# Separate care benefits from costs
sim_df['informal_care_benefits'] = sim_df['care_benefits_and_costs'].clip(lower=0)
sim_df['formal_care_costs_gov'] = -sim_df['care_benefits_and_costs'].clip(upper=0)

# Aggregate by care arrangement type
care_fiscal = sim_df.groupby(['age', 'choice']).agg({
    'informal_care_benefits': 'mean',
    'formal_care_costs_gov': 'mean',
})
```

**Pros**: Identifies care-specific fiscal impacts
**Cons**: Requires sign manipulation (error-prone)

#### Method 2: Care Transition Analysis

```python
# Fiscal impact of switching from informal to formal care
informal_care_rows = sim_df[sim_df['choice'].isin(INFORMAL_CARE_CHOICES)]
formal_care_rows = sim_df[sim_df['choice'].isin(FORMAL_CARE_CHOICES)]

# Compare fiscal impacts
fiscal_informal = informal_care_rows['net_government_budget'].mean()
fiscal_formal = formal_care_rows['net_government_budget'].mean()
fiscal_difference = fiscal_formal - fiscal_informal
```

**Pros**: Shows substitution effects
**Cons**: Cross-sectional comparison (not causal)

### 3.3 Labor Supply Fiscal Spillovers

#### Method 1: Employment Status Decomposition

```python
# Fiscal impact by employment status
employment_fiscal = sim_df.groupby(['age', 'is_retired']).agg({
    'total_tax_revenue': 'mean',
    'own_ssc': 'mean',
    'household_unemployment_benefits': 'mean',
})

# Calculate revenue loss from reduced labor supply
working_revenue = sim_df[sim_df['is_retired'] == False]['total_tax_revenue'].mean()
retired_revenue = sim_df[sim_df['is_retired'] == True]['total_tax_revenue'].mean()
revenue_loss = working_revenue - retired_revenue
```

**Pros**: Quantifies labor supply effects
**Cons**: Does not account for caregiving-specific labor supply reductions

#### Method 2: Caregiving Labor Supply Impact

```python
# Compare fiscal impact of working caregivers vs. non-working caregivers
caregiving_workers = sim_df[
    (sim_df['choice'].isin(INFORMAL_CARE_CHOICES)) &
    (sim_df['is_retired'] == False)
]
caregiving_non_workers = sim_df[
    (sim_df['choice'].isin(INFORMAL_CARE_CHOICES)) &
    (sim_df['is_retired'] == True)
]

# Fiscal difference
fiscal_workers = caregiving_workers['total_tax_revenue'].mean()
fiscal_non_workers = caregiving_non_workers['total_tax_revenue'].mean()
labor_supply_effect = fiscal_workers - fiscal_non_workers
```

**Pros**: Isolates caregiving-specific labor supply effects
**Cons**: Requires careful matching/controls

### 3.4 Present Value Calculations

#### Method 1: Simple NPV

```python
# Discount rate (e.g., 3% real)
r = 0.03
sim_df['discount_factor'] = (1 + r) ** (-sim_df['period'])

# Lifetime NPV per agent
lifetime_npv = (sim_df['net_government_budget'] * sim_df['discount_factor']).groupby('agent').sum()
average_lifetime_npv = lifetime_npv.mean()
```

#### Method 2: Age-Specific Discounting

```python
# Use age-specific discount rates (e.g., higher for older ages)
age_discount_rates = {age: 0.03 + (age - 30) * 0.001 for age in range(30, 101)}
sim_df['age_discount_rate'] = sim_df['age'].map(age_discount_rates)
sim_df['discount_factor'] = (1 + sim_df['age_discount_rate']) ** (-sim_df['period'])

# Lifetime NPV
lifetime_npv = (sim_df['net_government_budget'] * sim_df['discount_factor']).groupby('agent').sum()
```

**Pros**: More realistic discounting
**Cons**: Requires assumptions about age-specific rates

---

## 4. Missing Components: What Would Be Nice to Have

### 4.1 Disaggregated Care Benefits and Costs

**Current Issue**: `care_benefits_and_costs` is a net value (benefits - costs), making it difficult to:
- Separate informal care benefits from formal care costs
- Analyze substitution effects between care arrangements
- Calculate government expenditure on formal care separately

**Proposed Solution**: Return separate variables in `aux`:
```python
aux = {
    # ... existing variables ...
    "informal_care_benefits_light": annual_care_benefits_light / model_specs["wealth_unit"],
    "informal_care_benefits_intensive": annual_care_benefits_intensive / model_specs["wealth_unit"],
    "formal_care_costs_gov": annual_care_costs_weighted / model_specs["wealth_unit"],  # Positive value
    "care_benefits_and_costs": care_benfits_and_costs / model_specs["wealth_unit"],  # Keep for backward compatibility
}
```

**Benefits**:
- Clear separation of benefits vs. costs
- Easier decomposition of care-specific fiscal impacts
- Better analysis of substitution effects

### 4.2 Disaggregated SSC Components

**Current Issue**: `own_ssc` and `partner_ssc` are aggregates, making it difficult to:
- Separate pension/unemployment SSC from health/LTC SSC
- Analyze health insurance vs. pension system impacts
- Calculate LTC-specific contributions separately

**Proposed Solution**: Return component breakdown:
```python
aux = {
    # ... existing variables ...
    "own_ssc_pension_unempl": own_ssc_pension_unempl / model_specs["wealth_unit"],
    "own_ssc_health_ltc": own_ssc_health_ltc / model_specs["wealth_unit"],
    "partner_ssc_pension_unempl": partner_ssc_pension_unempl / model_specs["wealth_unit"],
    "partner_ssc_health_ltc": partner_ssc_health_ltc / model_specs["wealth_unit"],
    # Keep aggregates for backward compatibility
    "own_ssc": own_ssc / model_specs["wealth_unit"],
    "partner_ssc": partner_ssc / model_specs["wealth_unit"],
}
```

**Benefits**:
- Analyze health/LTC system separately from pension system
- Better understanding of LTC financing
- Policy analysis for specific social security components

### 4.3 Income Tax Decomposition

**Current Issue**: `income_tax` includes Soli and church tax, making it difficult to:
- Separate pure income tax from surcharges
- Analyze tax bracket distribution
- Calculate effective tax rates by income level

**Proposed Solution**: Return tax components:
```python
aux = {
    # ... existing variables ...
    "income_tax_pure": income_tax_pure / model_specs["wealth_unit"],
    "income_tax_soli": income_tax_soli / model_specs["wealth_unit"],
    "income_tax_church": income_tax_church / model_specs["wealth_unit"],
    "effective_tax_rate": income_tax_total / gross_household_income,
    "marginal_tax_rate": marginal_tax_rate,  # From tax bracket
    # Keep aggregate for backward compatibility
    "income_tax": income_tax_total / model_specs["wealth_unit"],
}
```

**Benefits**:
- Better tax policy analysis
- Effective vs. marginal tax rate analysis
- Distributional analysis by income level

### 4.4 Care Arrangement Indicators

**Current Issue**: Need to infer care arrangement from `choice`, making it error-prone to:
- Identify care arrangement types
- Calculate care-specific fiscal impacts
- Analyze care transitions

**Proposed Solution**: Add explicit indicators in `aux`:
```python
aux = {
    # ... existing variables ...
    "is_no_care": is_no_care(lagged_choice),
    "is_informal_care_light": is_light_informal_care(lagged_choice),
    "is_informal_care_intensive": is_intensive_informal_care(lagged_choice),
    "is_formal_care": is_formal_care(lagged_choice),
    "care_arrangement_type": care_arrangement_type,  # Categorical: 0=no_care, 1=informal_light, 2=informal_intensive, 3=formal
}
```

**Benefits**:
- Easier filtering and aggregation
- Clearer analysis of care-specific impacts
- Reduced risk of errors in care arrangement identification

### 4.5 Employment Status Indicators

**Current Issue**: Need to infer employment status from `choice`, making it difficult to:
- Calculate employment-specific fiscal impacts
- Analyze labor supply effects
- Compare working vs. non-working caregivers

**Proposed Solution**: Add explicit indicators in `aux`:
```python
aux = {
    # ... existing variables ...
    "was_worker": was_worker,
    "was_retired": was_retired,
    "was_unemployed": was_unemployed,
    "was_part_time": was_part_time,
    "was_full_time": was_full_time,
    "employment_status": employment_status,  # Categorical: 0=unemployed, 1=part_time, 2=full_time, 3=retired
}
```

**Benefits**:
- Easier employment status analysis
- Better labor supply fiscal impact calculations
- Clearer decomposition by employment type

### 4.6 Gross Household Income

**Current Issue**: `gross_labor_income` and `gross_retirement_income` are separate, making it difficult to:
- Calculate total gross household income
- Compute effective tax rates
- Analyze income distribution

**Proposed Solution**: Add aggregate:
```python
aux = {
    # ... existing variables ...
    "gross_household_income": (own_gross_income + partner_gross_income) / model_specs["wealth_unit"],
    "gross_household_income_own": own_gross_income / model_specs["wealth_unit"],
    "gross_household_income_partner": partner_gross_income / model_specs["wealth_unit"],
}
```

**Benefits**:
- Easier effective tax rate calculations
- Better income distribution analysis
- Clearer household income decomposition

### 4.7 Pension System Contributions and Benefits

**Current Issue**: Pension contributions (SSC) and benefits (pension payments) are not directly linked, making it difficult to:
- Analyze pension system sustainability
- Calculate net pension system impact (contributions - benefits)
- Analyze intergenerational transfers

**Proposed Solution**: Add pension-specific variables:
```python
aux = {
    # ... existing variables ...
    "pension_contributions_own": own_ssc_pension / model_specs["wealth_unit"],
    "pension_contributions_partner": partner_ssc_pension / model_specs["wealth_unit"],
    "pension_benefits_own": gross_retirement_income / model_specs["wealth_unit"],
    "pension_benefits_partner": gross_partner_pension / model_specs["wealth_unit"],
    "net_pension_system_impact": (pension_contributions_total - pension_benefits_total) / model_specs["wealth_unit"],
}
```

**Benefits**:
- Analyze pension system separately
- Better understanding of intergenerational transfers
- Policy analysis for pension reforms

### 4.8 Caregiving Leave Top-Up (for Counterfactuals)

**Current Issue**: In caregiving leave counterfactuals, top-up payments are included in `government_expenditures` but not separately identified, making it difficult to:
- Quantify cost of caregiving leave policies
- Compare different leave policy designs
- Analyze take-up effects

**Proposed Solution**: Add separate variable (already exists in counterfactual budget equations, but should be in `aux`):
```python
aux = {
    # ... existing variables ...
    "caregiving_leave_top_up": caregiving_leave_top_up / model_specs["wealth_unit"],
}
```

**Benefits**:
- Clear quantification of policy costs
- Better policy comparison
- Easier cost-benefit analysis

### 4.9 Formal Care Cost Breakdown

**Current Issue**: `formal_care_costs[period]` is a single value, but in reality, formal care costs may vary by:
- Care intensity (light vs. intensive ADL needs)
- Care arrangement (home care vs. institutional care)
- Government subsidy level

**Proposed Solution**: If model is extended to include care intensity in formal care:
```python
aux = {
    # ... existing variables ...
    "formal_care_costs_light": formal_care_costs_light / model_specs["wealth_unit"],
    "formal_care_costs_intensive": formal_care_costs_intensive / model_specs["wealth_unit"],
    "formal_care_subsidy_rate": formal_care_subsidy_rate,  # Government subsidy percentage
}
```

**Benefits**:
- More realistic formal care cost modeling
- Better analysis of care intensity effects
- Policy analysis for subsidy reforms

### 4.10 Wealth Tax / Inheritance Tax

**Current Issue**: Inheritance is included in `bequest_from_parent`, but inheritance tax is not calculated, making it difficult to:
- Analyze inheritance tax revenue
- Compare tax systems (income vs. wealth taxes)
- Analyze intergenerational wealth transfers

**Proposed Solution**: If inheritance tax is added to model:
```python
aux = {
    # ... existing variables ...
    "inheritance_tax": inheritance_tax / model_specs["wealth_unit"],
    "bequest_after_tax": bequest_after_tax / model_specs["wealth_unit"],
}
```

**Benefits**:
- Complete tax system analysis
- Better understanding of wealth transfers
- Policy analysis for inheritance tax reforms

---

## 5. Recommended Implementation Priority

### High Priority (Easy Wins)
1. **Disaggregated care benefits and costs** (Section 4.1)
   - Low implementation cost
   - High analytical value
   - Minimal code changes

2. **Care arrangement indicators** (Section 4.4)
   - Very low implementation cost
   - Reduces errors in analysis
   - High usability

3. **Employment status indicators** (Section 4.5)
   - Very low implementation cost
   - High analytical value
   - Already computed in `budget_constraint`

### Medium Priority (Moderate Effort)
4. **Disaggregated SSC components** (Section 4.2)
   - Moderate implementation cost
   - High analytical value for health/LTC analysis
   - Requires changes to `calc_government_budget_components`

5. **Gross household income** (Section 4.6)
   - Low implementation cost
   - High analytical value
   - Simple aggregation

6. **Income tax decomposition** (Section 4.3)
   - Moderate implementation cost
   - Requires changes to `calc_inc_tax_for_single_income`
   - High value for tax policy analysis

### Low Priority (Future Extensions)
7. **Pension system breakdown** (Section 4.7)
   - Requires model extensions
   - High value for pension policy analysis
   - May require separate pension system modeling

8. **Formal care cost breakdown** (Section 4.9)
   - Requires model extensions
   - Depends on care intensity modeling
   - Future research direction

9. **Inheritance tax** (Section 4.10)
   - Requires tax system extensions
   - May not be relevant for current research
   - Future policy consideration

---

## 6. Example Fiscal Analysis Workflow

### Step 1: Load Simulated Data
```python
import pandas as pd
import numpy as np

# Load baseline and counterfactual simulations
baseline_df = pd.read_pickle("baseline_simulation.pkl")
cf_df = pd.read_pickle("counterfactual_simulation.pkl")
```

### Step 2: Calculate Aggregate Fiscal Impacts
```python
# Aggregate by age
baseline_fiscal = baseline_df.groupby('age').agg({
    'net_government_budget': 'mean',
    'total_tax_revenue': 'mean',
    'government_expenditures': 'mean',
})

# Counterfactual comparison
fiscal_impact = cf_df.groupby('age')['net_government_budget'].mean() - \
                baseline_df.groupby('age')['net_government_budget'].mean()
```

### Step 3: Decompose Fiscal Impact
```python
# Revenue impact
revenue_impact = (cf_df['total_tax_revenue'] - baseline_df['total_tax_revenue']).groupby('age').mean()

# Expenditure impact
expenditure_impact = (cf_df['government_expenditures'] - baseline_df['government_expenditures']).groupby('age').mean()

# Net impact
net_impact = revenue_impact - expenditure_impact
```

### Step 4: Care-Specific Analysis
```python
# Separate care benefits and costs (if available)
if 'informal_care_benefits_light' in baseline_df.columns:
    care_benefits_impact = (cf_df['informal_care_benefits_light'] -
                           baseline_df['informal_care_benefits_light']).groupby('age').mean()
    formal_care_costs_impact = (cf_df['formal_care_costs_gov'] -
                               baseline_df['formal_care_costs_gov']).groupby('age').mean()
else:
    # Fallback: use aggregated variable
    care_net_impact = (cf_df['care_benefits_and_costs'] -
                      baseline_df['care_benefits_and_costs']).groupby('age').mean()
```

### Step 5: Lifetime NPV Calculation
```python
# Discount rate
r = 0.03

# Calculate lifetime NPV per agent
baseline_df['discount_factor'] = (1 + r) ** (-baseline_df['period'])
cf_df['discount_factor'] = (1 + r) ** (-cf_df['period'])

baseline_lifetime_npv = (baseline_df['net_government_budget'] *
                        baseline_df['discount_factor']).groupby('agent').sum()
cf_lifetime_npv = (cf_df['net_government_budget'] *
                   cf_df['discount_factor']).groupby('agent').sum()

# Average lifetime fiscal impact
avg_lifetime_impact = (cf_lifetime_npv - baseline_lifetime_npv).mean()
```

### Step 6: Visualization
```python
import matplotlib.pyplot as plt

# Plot fiscal impact by age
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Revenue and expenditure
axes[0].plot(revenue_impact.index, revenue_impact.values, label='Revenue Impact')
axes[0].plot(expenditure_impact.index, expenditure_impact.values, label='Expenditure Impact')
axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Fiscal Impact (€)')
axes[0].set_title('Fiscal Impact Decomposition by Age')
axes[0].legend()

# Net impact
axes[1].plot(net_impact.index, net_impact.values, label='Net Fiscal Impact')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Net Fiscal Impact (€)')
axes[1].set_title('Net Government Budget Impact by Age')
axes[1].legend()

plt.tight_layout()
plt.savefig('fiscal_impact_analysis.png')
```

---

## 7. Summary

### Current Capabilities
✅ Aggregate fiscal impacts (revenue, expenditures, net budget)
✅ Revenue decomposition (income tax, SSC)
✅ Expenditure decomposition (child benefits, care benefits, unemployment benefits)
✅ Counterfactual comparisons
✅ Lifetime NPV calculations

### Limitations
❌ Care benefits and costs are aggregated (hard to separate)
❌ SSC components are aggregated (pension vs. health/LTC)
❌ Income tax includes surcharges (hard to separate)
❌ No explicit care arrangement indicators
❌ No explicit employment status indicators

### Recommended Next Steps
1. Implement high-priority extensions (Sections 4.1, 4.4, 4.5)
2. Test fiscal analysis workflow with current data
3. Identify additional needs based on research questions
4. Implement medium-priority extensions as needed
5. Consider low-priority extensions for future research

---

**Document Version**: 1.0
**Last Updated**: 2024
**Author**: AI Assistant (based on codebase analysis)
