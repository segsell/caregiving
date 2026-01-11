# Analysis of Matched Differences Event Study Approach

## Overview

This document analyzes the matched differences event study approach used in `task_plot_labor_supply_differences_no_care_demand.py` and critically examines whether the event study style plots are correctly implemented.

## Methodology: Matched Differences Event Study

### Core Concept

The matched differences approach compares outcomes between two scenarios (baseline vs. counterfactual) for the **same agents at the same time periods**, then aligns observations by distance to a reference event (e.g., first caregiving spell).

### Step-by-Step Process

1. **Data Preparation**
   - Filter to alive agents (`health != DEAD`)
   - Optionally filter to ever-caregivers or ever-care-demand agents
   - Ensure `agent` and `period` are columns (not index levels)

2. **Outcome Calculation**
   - Calculate outcomes (work, ft, pt, etc.) for both scenarios
   - For baseline: uses `WORK`, `FULL_TIME`, `PART_TIME`, `INFORMAL_CARE` choice sets
   - For counterfactual: uses `WORK_NO_CARE_DEMAND`, `FULL_TIME_NO_CARE_DEMAND`, `PART_TIME_NO_CARE_DEMAND` choice sets
   - Additional outcomes: gross_labor_income, savings, wealth, savings_rate

3. **Matching and Difference Computation**
   - Merge on `(agent, period)` to ensure same agent at same time period
   - Compute differences: `diff_outcome = outcome_original - outcome_counterfactual`
   - This ensures we're comparing the same individual at the same point in their lifecycle

4. **Event Alignment**
   - Identify first caregiving spell in baseline data:
     - Find first period where `choice.isin(INFORMAL_CARE)`
     - Create `first_care_period` for each agent
   - Compute `distance_to_first_care = period - first_care_period`
   - This creates an event study timeline where t=0 is the start of first caregiving spell

5. **Aggregation**
   - Filter to window around event (e.g., -20 to +20 periods)
   - Group by `distance_to_first_care` and compute mean differences
   - This gives average treatment effect at each distance from the event

6. **Plotting**
   - Plot differences (baseline - counterfactual) on y-axis
   - Distance to first care on x-axis
   - Interpretation: positive values mean baseline has higher outcome than counterfactual

## Critical Examination of Implementation

### ✅ Correct Aspects

1. **Matching Logic**
   - ✅ Correctly merges on `(agent, period)` to ensure matched comparisons
   - ✅ Uses `how="inner"` to only include observations present in both scenarios
   - ✅ This ensures we're comparing the same agent at the same lifecycle stage

2. **Event Definition**
   - ✅ Correctly identifies first caregiving spell using `choice.isin(INFORMAL_CARE)`
   - ✅ Uses `drop_duplicates("agent")` to get first occurrence per agent
   - ✅ Sorts by `["agent", "period"]` before deduplication to ensure first period

3. **Distance Calculation**
   - ✅ Correctly computes `distance_to_first_care = period - first_care_period`
   - ✅ Negative values = periods before first care
   - ✅ Zero = period of first care
   - ✅ Positive values = periods after first care

4. **Difference Computation**
   - ✅ Correctly computes `diff = original - counterfactual`
   - ✅ For employment: positive diff means baseline has higher employment
   - ✅ This is the standard treatment effect interpretation

5. **Aggregation**
   - ✅ Groups by `distance_to_first_care` and takes mean
   - ✅ This averages across all agents at each distance
   - ✅ Correctly handles missing values (agents without first care get NaN distance)

### ⚠️ Potential Issues and Considerations

1. **Missing First Care Period**
   - **Issue**: Agents who never provide care will have `NaN` for `first_care_period`
   - **Current Handling**: These agents are excluded from the merged data after window filtering
   - **Assessment**: ✅ This is correct - we only want to analyze agents who actually provide care

2. **Window Trimming**
   - **Issue**: Window trimming happens AFTER merging, so agents with first care outside window are excluded
   - **Current Handling**: `merged = merged[(distance >= -window) & (distance <= window)]`
   - **Assessment**: ✅ This is correct - we want a symmetric window around the event

3. **Multiple Care Spells**
   - **Issue**: Agents may have multiple care spells, but only first is used for alignment
   - **Current Handling**: Uses `first_care_period` only
   - **Assessment**: ✅ This is correct for event study - we align by first event, not subsequent ones

4. **Sample Selection**
   - **Issue**: `ever_caregivers` and `ever_care_demand` filters affect which agents are included
   - **Current Handling**: Filters applied before matching
   - **Assessment**: ✅ This is correct - ensures we're comparing relevant populations

5. **Counterfactual Alignment**
   - **Issue**: Counterfactual (no care demand) may have different agent-period observations
   - **Current Handling**: Uses `inner` merge, so only common agent-periods are included
   - **Assessment**: ✅ This is correct - ensures matched comparison

6. **Age vs. Period Alignment**
   - **Issue**: Event study uses period-based distance, not age-based
   - **Current Handling**: `distance_to_first_care` is in periods, not ages
   - **Assessment**: ✅ This is correct for event study - we want time relative to event, not absolute age

### 🔍 Edge Cases to Consider

1. **Agents with First Care at Different Ages**
   - Agents who start caregiving at age 50 vs. age 60 are both aligned to t=0
   - This is correct for event study - we want to see effects relative to event timing
   - However, age-specific effects may be masked by this aggregation

2. **Pre-Event Differences**
   - Differences before t=0 (negative distances) should be close to zero if randomization is good
   - Non-zero pre-event differences may indicate selection bias or anticipation effects
   - Current implementation correctly captures these

3. **Post-Event Persistence**
   - Differences after t=0 show how effects persist over time
   - Current implementation correctly tracks this

## Conclusion

The matched differences event study implementation appears **correct** for its intended purpose:

- ✅ Properly matches agents across scenarios
- ✅ Correctly identifies and aligns by first caregiving event
- ✅ Computes differences appropriately
- ✅ Aggregates correctly for event study visualization

The approach is methodologically sound for examining how caregiving affects outcomes relative to the timing of first caregiving spell, averaged across all caregivers.

## Recommendations

1. **Consider Age-Specific Analysis**: While the overall event study is correct, consider splitting by age at first care to examine heterogeneity
2. **Pre-Event Diagnostics**: Check that pre-event differences are close to zero to validate the comparison
3. **Robustness Checks**: Consider alternative event definitions (e.g., first care demand vs. first caregiving)
4. **Standard Errors**: Current implementation shows means only - consider adding confidence intervals for statistical inference
