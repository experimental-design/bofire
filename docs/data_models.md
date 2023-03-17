# Data Models vs. Functional Components

Data models in bofire hold static data of an optimization problem. These are input and output features as well as constraints making up the domain. They further include possible optimization objectives, acquisition functions, and kernels.

All data models in ```bofire.data_models```, are specified as pydantic models and inherit from ```bofire.data_models.base.BaseModel```. These data models can be (de)serialized via ```.dict()``` and ```.json()``` (provided by pydantic). A json schema of each data model can be obtained using ```.schema()```.

For surrogates and strategies, all functional parts are located in ```bofire.surrogates``` and ```bofire.strategies```. These functionalities include the ```ask``` and ```tell``` as well as ```fit``` and ```predict``` methods. All class attributes (used by these method) are also removed from the data models. Each functional entity is initialized using the corresponding data model. As an example, consider the following data model of a ```RandomStrategy```: