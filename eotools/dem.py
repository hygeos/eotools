import xarray as xr
from core.process.blockwise import BlockProcessor
from core.tools import Var
from xarray import Dataset


class InitAltitude(BlockProcessor):
    """
    Altitude initialization

    Currently sets altitude to zero.
    """
    def input_vars(self) -> list[Var]:
        return [Var("latitude"), Var("longitude")]

    def created_vars(self) -> list[Var]:
        return [Var("altitude", attrs={"units": "m"})]

    def process_block(self, block: Dataset) -> None:
        block["altitude"] = xr.zeros_like(block["latitude"])
        block["altitude"].attrs.update(units='m')
