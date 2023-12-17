
import pickle
import PyPluMA
import PyIO
import torch
from torch.utils.data import DataLoader

class InferenceMapPlugin:
    def input(self, inputfile):
        self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
        dataset = open(PyPluMA.prefix()+"/"+self.parameters["dataset"], "rb")
        capri_dataset = pickle.load(dataset)
        modelfile = open(PyPluMA.prefix()+"/"+self.parameters["model"], "rb")
        model = pickle.load(modelfile)
        capri_loader = DataLoader(capri_dataset, batch_size=128, shuffle=False, pin_memory=False)

        from tqdm import tqdm
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device=torch.device("cpu")

# Infer in batches
        all_outputs = []
        with torch.no_grad():
          for grid, all_energies in tqdm(capri_loader):
            grid = grid.to(device)
            all_energies = all_energies.float().to(device)
            model = model.to(device)
            output, attn = model(grid, all_energies)
            all_outputs.append(output)
        output = torch.cat(all_outputs, axis=0)
        myoutfile = open(outputfile, "wb")
        pickle.dump(output, myoutfile)
