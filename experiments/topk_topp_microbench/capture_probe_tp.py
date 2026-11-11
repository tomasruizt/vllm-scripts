"""capture_probe.SolProbe that saves on TP rank 0 only, so TP ranks do not write the same files."""
from capture_probe import SolProbe


class RankZeroProbe(SolProbe):
    def capture_logits(self, logits, k, p) -> None:
        if self.rank == 0:
            super().capture_logits(logits, k, p)
