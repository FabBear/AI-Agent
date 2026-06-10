"""Common API response envelope."""

from typing import Generic, TypeVar

from pydantic import BaseModel

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    success: bool = True
    data: DataT


def success(data: DataT) -> ApiResponse[DataT]:
    return ApiResponse(data=data)
