from bofire.data_models.priors.api import BOTORCH_LENGTHCALE_PRIOR
from bofire.plot.api import plot_prior_pdf_plotly


def test_plot_prior_pdf_plotly():
    plot_prior_pdf_plotly(BOTORCH_LENGTHCALE_PRIOR(), lower=0, upper=10)
