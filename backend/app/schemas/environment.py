"""The 'hyper-local' half of the input — what the farmer tells us about their field."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SoilType(str, Enum):
    ALLUVIAL = "alluvial"
    BLACK = "black"
    RED = "red"
    LATERITE = "laterite"
    SANDY = "sandy"
    LOAM = "loam"
    CLAY = "clay"
    UNKNOWN = "unknown"


class Season(str, Enum):
    KHARIF = "kharif"  # monsoon
    RABI = "rabi"  # winter
    ZAID = "zaid"  # summer
    UNKNOWN = "unknown"


class IrrigationType(str, Enum):
    RAINFED = "rainfed"
    FLOOD = "flood"
    DRIP = "drip"
    SPRINKLER = "sprinkler"
    UNKNOWN = "unknown"


class EnvironmentContext(BaseModel):
    """Local conditions. Every field is optional — an empty context is valid and the
    advisory simply says less about siting and timing."""

    region: str | None = Field(None, description="District / state / agro-climatic zone")
    soil_type: SoilType = SoilType.UNKNOWN
    season: Season = Season.UNKNOWN
    irrigation: IrrigationType = IrrigationType.UNKNOWN

    rainfall_last_7d_mm: float | None = Field(None, ge=0, le=2000)
    temperature_c: float | None = Field(None, ge=-20, le=60)
    relative_humidity_pct: float | None = Field(None, ge=0, le=100)
    soil_ph: float | None = Field(None, ge=2.0, le=11.0)

    crop_stage: str | None = Field(None, description="e.g. vegetative, flowering, fruiting")
    days_since_last_spray: int | None = Field(None, ge=0, le=365)
    field_area_ha: float | None = Field(None, gt=0, le=10_000)
    organic_only: bool = Field(
        False, description="If true, the plan must avoid synthetic chemical actives."
    )

    def to_prompt_lines(self) -> list[str]:
        """Only the facts we actually have — never pad the prompt with 'unknown'."""
        pairs = [
            ("Region", self.region),
            ("Soil type", None if self.soil_type is SoilType.UNKNOWN else self.soil_type.value),
            ("Season", None if self.season is Season.UNKNOWN else self.season.value),
            (
                "Irrigation",
                None if self.irrigation is IrrigationType.UNKNOWN else self.irrigation.value,
            ),
            (
                "Rainfall (last 7 days)",
                None if self.rainfall_last_7d_mm is None else f"{self.rainfall_last_7d_mm:.0f} mm",
            ),
            ("Temperature", None if self.temperature_c is None else f"{self.temperature_c:.0f} C"),
            (
                "Relative humidity",
                None
                if self.relative_humidity_pct is None
                else f"{self.relative_humidity_pct:.0f} %",
            ),
            ("Soil pH", None if self.soil_ph is None else f"{self.soil_ph:.1f}"),
            ("Crop stage", self.crop_stage),
            (
                "Days since last spray",
                None if self.days_since_last_spray is None else str(self.days_since_last_spray),
            ),
            ("Field area", None if self.field_area_ha is None else f"{self.field_area_ha:g} ha"),
        ]
        lines = [f"- {label}: {value}" for label, value in pairs if value]
        if self.organic_only:
            lines.append("- Constraint: ORGANIC ONLY — no synthetic chemical actives permitted.")
        return lines or ["- (no local context supplied)"]

    def wetness_note(self) -> str | None:
        """Rain drives fungal pressure and washes off contact sprays — say so explicitly."""
        rain = self.rainfall_last_7d_mm
        if rain is None:
            return None
        if rain >= 100:
            return (
                "Heavy recent rainfall (>=100 mm/7d): high fungal/bacterial pressure, "
                "contact sprays will wash off — favour systemic actives and drainage work."
            )
        if rain >= 30:
            return (
                "Moderate recent rainfall (30-100 mm/7d): elevated fungal pressure, "
                "time any spray to a 6+ hour dry window."
            )
        if rain <= 5:
            return (
                "Dry week (<=5 mm/7d): low splash-dispersal risk; check irrigation "
                "and mite/abiotic stress as alternative causes."
            )
        return None
