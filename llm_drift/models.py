from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator


class Probe(BaseModel):
    id: str
    prompt: str
    assertions: List[str] = Field(default_factory=list)


class ProbeSuite(BaseModel):
    name: str
    model: str
    provider: str = "openai"
    probes: List[Probe] = Field(min_length=1)

    @field_validator("probes")
    @classmethod
    def probe_ids_unique(cls, v: List[Probe]) -> List[Probe]:
        ids = [p.id for p in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"probe ids must be unique; duplicates: {sorted(dupes)}")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProbeSuite:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
