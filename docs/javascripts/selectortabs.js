// set default strategy
let problemDescription = new Map([
    ["objective", "single-objective"],
    ["tasks", "single-task"],
    ["domain", "simple-domain"],
])

const strategyMap = new Map([
    ["single-objective__single-task__simple-domain", ["SoboStrategy", ""]],
    ["single-objective__single-task__many-categorical-features", ["EntingStrategy", ""]],
    ["single-objective__multi-task__simple-domain", ["SoboStrategy", "MultiTaskGPSurrogate"]],
    ["single-objective__multi-fidelity__simple-domain", ["MultiFidelity", ""]],

    ["multi-objective__single-task__simple-domain", ["MoboStrategy", ""]],
])

const getSurrogateCode = (surrogateDataModel) => {
    if (surrogateDataModel === "") {
        return ""
    }
    return `
# define the surrogate data model
surrogate_data_model = BotorchSurrogates(surrogates=[
    ${surrogateDataModel}(
        inputs=domain.inputs,
        outputs=domain.outputs,
    )
])
`}

const getStrategyComment = (strategyDataModel) => {
    return (strategyDataModel !== "EntingStrategy") ? "" : `
# the default GP surrogate is slow to optimize when many discrete features are present
# the ENTMOOT model can optimize over domains with many categories`
}

const getStrategyCode = () => {
    const dataModels = strategyMap.get([...problemDescription.values()].join("__"))
    if (dataModels === undefined) {
        return hljs.highlight("# There isn't currently a BoFire model that works for this problem.", {language: "python"})
    }

    const [strategyDataModel, surrogateDataModel] = dataModels
    const requiresSurrogate = (surrogateDataModel !== "")
    const surrogateImport = (!requiresSurrogate) ? "" : `
from bofire.data_models.surrogates.api import ${surrogateDataModel}`

    const strategyComment = getStrategyComment(strategyDataModel)
    const surrogateCode = getSurrogateCode(surrogateDataModel)

    return hljs.highlight(
`from bofire.data_models.strategies.api import ${strategyDataModel} ${surrogateImport}
import bofire.strategies.api as strategies
${surrogateCode}
# define the strategy data model ${strategyComment}
strategy_data_model = ${strategyDataModel}(
    domain=domain${!requiresSurrogate ? "" : `
    surrogate=surrogate_data_model`}
)

# create an instance of the functional strategy
# to understand the difference between data models and functional components,
# see https://experimental-design.github.io/bofire/data_models_functionals/
strategy = strategies.map(strategy_data_model)
`, { language : "python"}
    )
}

const updateStrategyCodeBlock = () => {
    const codeBlock = document.getElementById("stategyTemplate")
    codeBlock.innerHTML = getStrategyCode().value
}

const tabSync = () => {
    const tabs = document.querySelectorAll(".tabbed-set > input")
    for (const tab of tabs) {
      tab.addEventListener("click", () => {
        const current = document.querySelector(`label[for=${tab.id}]`)
        const pos = current.getBoundingClientRect().top
        // const labels = document.querySelectorAll('.tabbed-set > label, .tabbed-alternate > .tabbed-labels > label')
        
        const updatedGroup = tab.parentElement.id
        // const updatedGroup = selectorTabStrategies[tab.id]
        problemDescription.set(updatedGroup, tab.id)
        updateStrategyCodeBlock()

        // Preserve scroll position
        const delta = (current.getBoundingClientRect().top) - pos
        window.scrollBy(0, delta)
      })
    }
  }

document$.subscribe(function() {
    ["objective", "tasks", "domain"].forEach(k => {
        document.querySelector(`#${k}-marker`).nextElementSibling.setAttribute("id", k)
    })
    tabSync()
    updateStrategyCodeBlock()
})
