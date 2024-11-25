import streamlit as st
import pymc as pm
import matplotlib.pyplot as plt

st.title("Bayesian Inference for ANCA-associated Vasculitis (AAV)")

st.sidebar.header("Input Variables")

# Sidebar inputs
observed_incidence = st.sidebar.slider("Observed Incidence (per 100,000/year)", 0.1, 20.0, 2.5, 0.1)
observed_prevalence = st.sidebar.slider("Observed Prevalence (per 100,000)", 0.1, 100.0, 30.0, 0.1)
observed_mortality = st.sidebar.slider("Observed Mortality (fraction)", 0.01, 0.3, 0.1, 0.01)
observed_relapse_rate = st.sidebar.slider("Observed Relapse Rate (fraction)", 0.01, 0.5, 0.2, 0.01)
observed_treatment_effect = st.sidebar.slider("Observed Treatment Effect (fraction)", 0.1, 0.9, 0.5, 0.1)

# Run Bayesian Inference when button is clicked
if st.sidebar.button("Run Bayesian Inference"):
    with pm.Model() as model:
        # Priors
        incidence = pm.Uniform("incidence", lower=0.1, upper=10)
        prevalence = pm.Uniform("prevalence", lower=1, upper=50)
        mortality = pm.Uniform("mortality", lower=0.01, upper=0.3)
        relapse_rate = pm.Uniform("relapse_rate", lower=0.01, upper=0.5)
        treatment_effect = pm.Uniform("treatment_effect", lower=0.1, upper=0.9)

        # Likelihoods
        observed_inc = pm.Normal("observed_incidence", mu=incidence, sigma=1, observed=observed_incidence)
        observed_prev = pm.Normal("observed_prevalence", mu=prevalence, sigma=5, observed=observed_prevalence)
        observed_mort = pm.Normal("observed_mortality", mu=mortality, sigma=0.05, observed=observed_mortality)
        observed_relapse = pm.Normal("observed_relapse_rate", mu=relapse_rate, sigma=0.1, observed=observed_relapse_rate)
        observed_treat_eff = pm.Normal("observed_treatment_effect", mu=treatment_effect, sigma=0.1, observed=observed_treatment_effect)

        # Sampling
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    st.success("Bayesian inference completed!")

    # Plot posterior distributions
    st.header("Posterior Distributions")
    fig, axes = plt.subplots(5, 1, figsize=(8, 20))
    pm.plot_posterior(
        trace,
        var_names=["incidence", "prevalence", "mortality", "relapse_rate", "treatment_effect"],
        ax=axes
    )
    plt.tight_layout()
    st.pyplot(fig)