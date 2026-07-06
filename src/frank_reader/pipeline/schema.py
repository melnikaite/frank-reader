from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    original: str
    translation: str


class TextBlock(BaseModel):
    order: int
    type: Literal["phrase", "heading", "list_item", "caption"]
    original: str
    translation: str | None = None
    chunks: list[Chunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_by_type(self):
        if self.type == "phrase" and not self.chunks:
            # Small models reliably emit chunk-less phrases with a whole-phrase
            # translation for degenerate lines like "Art 74a (weggefallen)".
            # Rejecting the whole page over that is worse than accepting a
            # single chunk covering the entire phrase.
            if self.translation:
                self.chunks = [Chunk(original=self.original, translation=self.translation)]
            else:
                raise ValueError("phrase must have chunks")
        if self.type != "phrase" and not self.translation:
            raise ValueError(f"{self.type} must have translation")
        return self


class ImageLabel(BaseModel):
    original: str
    translation: str


class ImageAnnotation(BaseModel):
    image_ref: int
    labels: list[ImageLabel] = Field(default_factory=list)


class Term(BaseModel):
    term: str
    translation: str


class PageResult(BaseModel):
    page_summary: str
    detected_language: str
    text_blocks: list[TextBlock]
    image_annotations: list[ImageAnnotation] = Field(default_factory=list)
    new_terms: list[Term] = Field(default_factory=list)


class ImageLabelsResult(BaseModel):
    labels: list[ImageLabel] = Field(default_factory=list)
