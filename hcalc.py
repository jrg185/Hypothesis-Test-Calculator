import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_one_sample_t_test(sample_size, sample_mean, sample_std, hypothesized_mean, alpha, test_type, alternative):
    t_statistic = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(sample_size))
    df = sample_size - 1
    
    if test_type == "Two-tailed":
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    else:  # One-tailed
        if alternative == "Greater":
            p_value = 1 - stats.t.cdf(t_statistic, df)
        else:  # Less
            p_value = stats.t.cdf(t_statistic, df)
    
    reject_null = p_value < alpha
    return t_statistic, p_value, reject_null, df

def calculate_two_sample_t_test(n1, mean1, std1, n2, mean2, std2, alpha, test_type, alternative):
    df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
    t_statistic = (mean1 - mean2) / np.sqrt(std1**2/n1 + std2**2/n2)
    
    if test_type == "Two-tailed":
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    else:  # One-tailed
        if alternative == "Greater":
            p_value = 1 - stats.t.cdf(t_statistic, df)
        else:  # Less
            p_value = stats.t.cdf(t_statistic, df)
    
    reject_null = p_value < alpha
    return t_statistic, p_value, reject_null, df

def calculate_paired_t_test(n, mean_diff, std_diff, alpha, test_type, alternative):
    t_statistic = mean_diff / (std_diff / np.sqrt(n))
    df = n - 1
    
    if test_type == "Two-tailed":
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    else:  # One-tailed
        if alternative == "Greater":
            p_value = 1 - stats.t.cdf(t_statistic, df)
        else:  # Less
            p_value = stats.t.cdf(t_statistic, df)
    
    reject_null = p_value < alpha
    return t_statistic, p_value, reject_null, df

def calculate_one_sample_z_test(sample_size, sample_mean, population_std, hypothesized_mean, alpha, test_type, alternative):
    z_statistic = (sample_mean - hypothesized_mean) / (population_std / np.sqrt(sample_size))
    
    if test_type == "Two-tailed":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    else:  # One-tailed
        if alternative == "Greater":
            p_value = 1 - stats.norm.cdf(z_statistic)
        else:  # Less
            p_value = stats.norm.cdf(z_statistic)
    
    reject_null = p_value < alpha
    return z_statistic, p_value, reject_null

def calculate_chi_square_goodness_of_fit(observed, expected, alpha):
    chi2_statistic, p_value = stats.chisquare(observed, expected)
    df = len(observed) - 1
    reject_null = p_value < alpha
    return chi2_statistic, p_value, reject_null, df

# Set up the Streamlit app
st.title("Comprehensive Hypothesis Test Calculator")

# Test selection
test_type = st.selectbox("Select Test Type", [
    "One Sample T-Test",
    "Two Sample T-Test",
    "Paired T-Test",
    "One Sample Z-Test",
    "Chi-Square Goodness of Fit Test"
])

# Common inputs
alpha = st.number_input("Significance Level (α)", min_value=0.01, max_value=0.99, value=0.05, step=0.01)
tail_type = st.radio("Test Type", ("Two-tailed", "One-tailed"))

if tail_type == "One-tailed":
    alternative = st.radio("Alternative Hypothesis", ("Greater", "Less"))
else:
    alternative = None

# Test-specific inputs
if test_type == "One Sample T-Test":
    sample_size = st.number_input("Sample Size", min_value=2, step=1, value=30)
    sample_mean = st.number_input("Sample Mean", value=0.0, step=0.1)
    sample_std = st.number_input("Sample Standard Deviation", min_value=0.0001, value=1.0, step=0.1)
    hypothesized_mean = st.number_input("Hypothesized Mean", value=0.0, step=0.1)

elif test_type == "Two Sample T-Test":
    n1 = st.number_input("Sample Size (Group 1)", min_value=2, step=1, value=30)
    mean1 = st.number_input("Sample Mean (Group 1)", value=0.0, step=0.1)
    std1 = st.number_input("Sample Standard Deviation (Group 1)", min_value=0.0001, value=1.0, step=0.1)
    n2 = st.number_input("Sample Size (Group 2)", min_value=2, step=1, value=30)
    mean2 = st.number_input("Sample Mean (Group 2)", value=0.0, step=0.1)
    std2 = st.number_input("Sample Standard Deviation (Group 2)", min_value=0.0001, value=1.0, step=0.1)

elif test_type == "Paired T-Test":
    n = st.number_input("Number of Pairs", min_value=2, step=1, value=30)
    mean_diff = st.number_input("Mean of Differences", value=0.0, step=0.1)
    std_diff = st.number_input("Standard Deviation of Differences", min_value=0.0001, value=1.0, step=0.1)

elif test_type == "One Sample Z-Test":
    sample_size = st.number_input("Sample Size", min_value=2, step=1, value=30)
    sample_mean = st.number_input("Sample Mean", value=0.0, step=0.1)
    population_std = st.number_input("Population Standard Deviation", min_value=0.0001, value=1.0, step=0.1)
    hypothesized_mean = st.number_input("Hypothesized Mean", value=0.0, step=0.1)

elif test_type == "Chi-Square Goodness of Fit Test":
    num_categories = st.number_input("Number of Categories", min_value=2, step=1, value=3)
    observed = [st.number_input(f"Observed Frequency {i+1}", min_value=0, step=1, value=10) for i in range(num_categories)]
    expected = [st.number_input(f"Expected Frequency {i+1}", min_value=0.0001, step=1.0, value=10.0) for i in range(num_categories)]

# Calculate button
if st.button("Calculate"):
    if test_type == "One Sample T-Test":
        result = calculate_one_sample_t_test(sample_size, sample_mean, sample_std, hypothesized_mean, alpha, tail_type, alternative)
        statistic, p_value, reject_null, df = result
        statistic_name = "T-statistic"
    elif test_type == "Two Sample T-Test":
        result = calculate_two_sample_t_test(n1, mean1, std1, n2, mean2, std2, alpha, tail_type, alternative)
        statistic, p_value, reject_null, df = result
        statistic_name = "T-statistic"
    elif test_type == "Paired T-Test":
        result = calculate_paired_t_test(n, mean_diff, std_diff, alpha, tail_type, alternative)
        statistic, p_value, reject_null, df = result
        statistic_name = "T-statistic"
    elif test_type == "One Sample Z-Test":
        result = calculate_one_sample_z_test(sample_size, sample_mean, population_std, hypothesized_mean, alpha, tail_type, alternative)
        statistic, p_value, reject_null = result
        df = None
        statistic_name = "Z-statistic"
    elif test_type == "Chi-Square Goodness of Fit Test":
        result = calculate_chi_square_goodness_of_fit(observed, expected, alpha)
        statistic, p_value, reject_null, df = result
        statistic_name = "Chi-square statistic"
    
    # Display results
    st.write(f"{statistic_name}: {statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    st.write(f"Significance Level (α): {alpha}")
    
    if reject_null:
        st.write("Result: Reject the null hypothesis")
        if test_type != "Chi-Square Goodness of Fit Test":
            if tail_type == "Two-tailed":
                st.write("There is significant evidence of a difference.")
            else:
                if alternative == "Greater":
                    st.write("There is significant evidence of a greater value.")
                else:
                    st.write("There is significant evidence of a lesser value.")
        else:
            st.write("There is significant evidence that the observed frequencies differ from the expected frequencies.")
    else:
        st.write("Result: Fail to reject the null hypothesis")
        st.write("There is not enough evidence to conclude a significant difference.")

    # Visualization
    if test_type != "Chi-Square Goodness of Fit Test":
        fig, ax = plt.subplots()
        if test_type == "One Sample Z-Test":
            x = np.linspace(-4, 4, 100)
            y = stats.norm.pdf(x)
            ax.plot(x, y, 'b-', lw=2, label='Standard Normal Distribution')
            ax.axvline(statistic, color='r', linestyle='--', label='Z-statistic')
        else:
            x = np.linspace(stats.t.ppf(0.001, df), stats.t.ppf(0.999, df), 100)
            y = stats.t.pdf(x, df)
            ax.plot(x, y, 'b-', lw=2, label='t-distribution')
            ax.axvline(statistic, color='r', linestyle='--', label='T-statistic')
        
        if tail_type == "Two-tailed":
            ax.axvline(stats.t.ppf(alpha/2, df), color='g', linestyle='--', label=f'-t_{{α/2}}')
            ax.axvline(stats.t.ppf(1-alpha/2, df), color='g', linestyle='--', label=f't_{{α/2}}')
        else:
            if alternative == "Greater":
                ax.axvline(stats.t.ppf(1-alpha, df), color='g', linestyle='--', label=f't_{{α}}')
            else:
                ax.axvline(stats.t.ppf(alpha, df), color='g', linestyle='--', label=f'-t_{{α}}')
        
        ax.set_title(f"Distribution with Test Statistic and Critical Value(s)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.legend()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        x = np.linspace(0, stats.chi2.ppf(0.999, df), 100)
        y = stats.chi2.pdf(x, df)
        ax.plot(x, y, 'b-', lw=2, label='Chi-square distribution')
        ax.axvline(statistic, color='r', linestyle='--', label='Chi-square statistic')
        ax.axvline(stats.chi2.ppf(1-alpha, df), color='g', linestyle='--', label=f'Critical value')
        ax.set_title(f"Chi-square Distribution with Test Statistic and Critical Value")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.legend()
        st.pyplot(fig)

st.write("Note: This calculator assumes the necessary assumptions for each test are met (e.g., normality, independence).")