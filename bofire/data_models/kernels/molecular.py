from typing import Literal, Optional

from pydantic import model_validator

from bofire.data_models.features.molecular import CategoricalMolecularInput
from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.utils.cheminformatics import mutual_tanimoto_similarities


class MolecularKernel(FeatureSpecificKernel):
    pass


class TanimotoKernel(MolecularKernel):
    type: Literal["TanimotoKernel"] = "TanimotoKernel"
    ard: bool = True
    pre_compute_similarities: bool = False

    # private attributes, for pre-computation of similarities: will be overridden by tanimoto_gp, or auto-computed
    _fingerprint_settings_for_similarities: Optional[dict[str, Fingerprints]] = None
    _molecular_inputs: list[CategoricalMolecularInput] = None

    @model_validator(mode="after")
    def compute_mutual_tanimoto_distances(self):
        if not self.pre_compute_similarities or (
            self._computed_mutual_similarities is not None
        ):
            return self

        assert (
            self._molecular_inputs is not None
        ), "need molecular inputs ofr pre-computed distances"

        # fill fingerprint settings
        if self._fingerprint_settings_for_similarities is None:
            self._fingerprint_settings_for_similarities = {}
        for inp_ in self._molecular_inputs:
            if inp_.key not in list(self._fingerprint_settings_for_similarities):
                self._fingerprint_settings_for_similarities[inp_.key] = Fingerprints()