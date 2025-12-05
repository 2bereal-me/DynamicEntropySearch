from ms_entropy import FlashEntropySearchCoreForDynamicIndexing
import numpy as np

class DynamicWithFlash(FlashEntropySearchCoreForDynamicIndexing):
    def build_index(self, peak_data, max_indexed_mz, peak_data_nl=None):
        """
        :param peak_data_nl: Only used when building neutral loss index.

        """
        # Record the m/z, intensity, and spectrum index information for product ions.
        all_ions_mz = peak_data["ion_mz"]
        all_ions_intensity = peak_data["intensity"]
        all_ions_spec_idx = peak_data["spec_idx"]

        # Build index for fast access to the ion's m/z.
        all_ions_mz_idx_start = self._generate_index(all_ions_mz, max_indexed_mz)
        fragment_ion_index = (
            all_ions_mz_idx_start,
            all_ions_mz,
            all_ions_intensity,
            all_ions_spec_idx,
        )

        if peak_data_nl is not None:

            # Build the index for fast access to the neutral loss mass.
            all_nl_mass_idx_start = self._generate_index(peak_data_nl["nl_mass"], max_indexed_mz)
            neutral_loss_index = (
                all_nl_mass_idx_start,
                peak_data_nl["nl_mass"],
                peak_data_nl["intensity"],
                peak_data_nl["spec_idx"],
                peak_data_nl["ion_mz"],  # The fragment ion m/z corresponding to the neutral loss mass.
            )
        else:
            neutral_loss_index = None

        self.index = fragment_ion_index, neutral_loss_index
        return self.index

    def get_topn_spec_idx_and_similarity(
        self,
        similarity_array,
        topn=None,
        min_similarity=0.1,
    ):
        if topn == None:
            topn = len(similarity_array)

        if min_similarity == None:
            min_similarity = 0.0

        # Get topn indices
        topn_indices = np.argsort(similarity_array)[::-1][:topn]

        result = []
        result_idx = []
        for index in topn_indices:
            if similarity_array[index] < min_similarity:
                break

            selected_similarity = similarity_array[index]
            result.append(selected_similarity)
            result_idx.append(index)

        return result_idx, result
