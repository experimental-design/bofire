from bofire.data_models.llm.formulator import FormulatorConfig
from bofire.data_models.llm.provider import AzureOpenAILLMProvider
from bofire.llm.formulator import formulate

from dotenv import load_dotenv; load_dotenv()

config = FormulatorConfig(
    llm=AzureOpenAILLMProvider(
        model="gpt4o",
        api_key_env_var="AZURE_OPENAI_API_KEY",
        azure_endpoint_env_var="AZURE_OPENAI_ENDPOINT",
    ),
)

result = formulate(
    config,
    # "Using the experimental data we already collected, find the temperature "
    # "and catalyst concentration that maximize production yield, while keeping "
    # "the combined operating load under 100."
    # "I want to optimize the temperature of my process, it can be between 0 and 180 degrees celcius. I want the best yield.")
    "I want to optimize reaction time and the temperature of my process, it can be between 0 and 180 degrees celcius. I want the best yield I want a tradeoff with reaction time")
# result.classification → "single_objective"
# result.reasoning → why it classified this way
# result.domain → a real bofire Domain object (or None if not_experimental)
print(result.domain_spec) #→ the intermediate DomainSpec the LLM produced