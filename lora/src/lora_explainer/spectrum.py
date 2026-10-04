import torch


def singular_values(m: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(m)


def truncate(m: torch.Tensor, r: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    return u[:, :r] @ torch.diag(s[:r]) @ vh[:r]


def truncation_error(s: torch.Tensor, r: int) -> float:
    return s[r:].square().sum().sqrt().item()


def relative_error(s: torch.Tensor, r: int) -> float:
    return truncation_error(s, r) / s.square().sum().sqrt().item()


def energy_kept(s: torch.Tensor, r: int) -> float:
    return (s[:r].square().sum() / s.square().sum()).item()


def rank_for_energy(s: torch.Tensor, threshold: float) -> int:
    cumulative = s.square().cumsum(dim=0) / s.square().sum()
    reached = torch.searchsorted(cumulative, torch.tensor(threshold, dtype=cumulative.dtype))
    return min(int(reached) + 1, len(s))
