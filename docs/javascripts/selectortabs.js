const selectorTabStrategies = {
    // map from tab labels to group names
    "single-objective": "objective",
    "multi-objective": "objective",
    "single-task": "tasks",
    "multi-task": "tasks",
    "multi-fidelity": "tasks",
    "simple-domain": "domain",
    "many-categorical-features": "domain",
}

let strategies = {
    "objective": "single-objective",
    "tasks": "single-task",
    "domain": "simple-domain",
}

const getStrategyCode = () => {
    const objective = strategies["objective"];
    const tasks = strategies["tasks"];
    const domain = strategies["domain"];

    let strategyDataModel = ""
    if (objective == "single-objective") {
        if (tasks == "single-task") {
            if (domain == "simple-domain") {
                strategyDataModel = "SoboStrategy"
            } else if (domain == "many-categorical-features") {
                strategyDataModel = "EntingStrategy"
            }
        } else if (tasks == "multi-task") {
            strategyDataModel = "SoboStrategy"
        } else if (tasks == "multi-fidelity") {
            strategyDataModel = "MultiFidelityStrategy"
            // we define multi fidelity as a multi-task where you can query the other fidelities
        }
    } else if (objective == "multi-objective") {
        if (tasks == "single-task" && domain == "simple-domain") {
            strategyDataModel = "MoboStrategy"
        }
    }

    return hljs.highlight(
`# import the data model
from bofire.data_models.strategies.api import ${strategyDataModel}
import bofire.strategies.api as strategies

surrogate_data_model = BotorchSurrogates(surrogates=[
    MultiTaskGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
    )
])

strategy_data_model = ${strategyDataModel}(
    domain=domain
    surrogate=surrogate_data_model
)
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
        console.log(tab.id)
        const pos = current.getBoundingClientRect().top
        // const labels = document.querySelectorAll('.tabbed-set > label, .tabbed-alternate > .tabbed-labels > label')
        
        const updatedGroup = selectorTabStrategies[tab.id]
        strategies[updatedGroup] = tab.id
        updateStrategyCodeBlock()

        // Preserve scroll position
        const delta = (current.getBoundingClientRect().top) - pos
        window.scrollBy(0, delta)
      })
    }
  }

document$.subscribe(function() {
    console.log(document.title)
    tabSync()
})
