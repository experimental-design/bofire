from typing import Literal

from pydantic import model_validator

from bofire.data_models.features.molecular import CategoricalMolecularInput
from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.utils.cheminformatics import mutual_tanimoto_distances

class MolecularKernel(FeatureSpecificKernel):
    pre_compute_similarities: bool = False

class TanimotoKernel(MolecularKernel):
    type: Literal["TanimotoKernel"] = "TanimotoKernel"
    ard: bool = True
    _molecular_inputs: list[CategoricalMolecularInput] = None  # needed for pre-computation of tanimoto distances
    _computed_mutual_similarities: dict[str, list[float]] = None


    @model_validator(mode="after")
    def compute_mutual_tanimoto_distances(self):

        if not self.pre_compute_similarities or (self._computed_mutual_similarities is not None):
            return self

        assert self._molecular_inputs is not None, "need molecular inputs ofr pre-computed distances"

        self._computed_mutual_similarities = {}
        for inp in self._molecular_inputs:
            print(f"computing tanimoto distances for input {inp.key:}")
            self._computed_mutual_similarities[inp.key] = mutual_tanimoto_distances(inp.categories)