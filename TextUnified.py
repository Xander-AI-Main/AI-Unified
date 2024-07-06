from TextModel import TextModel
import json

def returnArch (data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if  i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]

arch_data = {}
task = "text"
mainType='topic classification'
archType='default'

with open ('arch.json', 'r') as f:
    arch_data = json.load(f)

architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)

model = TextModel(
    dataset_url='train.csv',
    hasChanged=False,
    task='text',
    mainType='topic classification',
    archType='Default',
    architecture=architecture,
    hyperparameters=hyperparameters
)

model_result = model.execute()
print(model_result)