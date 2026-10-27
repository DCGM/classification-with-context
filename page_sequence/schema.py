"""Versioned metadata and model contracts. No source text or OCR enters this package."""
from dataclasses import dataclass

PAGE_TYPES = (
    'Abstract', 'Advertisement', 'Appendix', 'BackCover', 'BackEndPaper', 'BackEndSheet',
    'Bibliography', 'Blank', 'CalibrationTable', 'Colophon', 'Cover', 'CustomInclude',
    'Dedication', 'Edge', 'Errata', 'FlyLeaf', 'FragmentsOfBookbinding', 'FrontCover',
    'FrontEndPaper', 'FrontEndSheet', 'FrontJacket', 'Frontispiece', 'Illustration',
    'Impressum', 'Imprimatur', 'Index', 'Jacket', 'ListOfIllustrations', 'ListOfMaps',
    'ListOfTables', 'Map', 'NormalPage', 'Obituary', 'Preface', 'SheetMusic', 'Spine',
    'Table', 'TableOfContents', 'TitlePage',
)
PLACEMENTS = ('left', 'right', 'single')
LABEL2ID = {name: i for i, name in enumerate(PAGE_TYPES)}
IGNORE = -100
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Page:
    library: str
    doc: str
    page: str
    order: int
    path: str = ''
    db_type: str = ''
    annotated_type: str = ''
    placement: str = ''
    pagenum: str = ''

    @property
    def key(self):
        return f'{self.library}_{self.page}'

    @property
    def doc_key(self):
        return self.library, self.doc

    def target(self):
        # Unsupported curated labels are rejected on ingestion, never replaced by weak labels.
        gold = bool(self.annotated_type)
        label = LABEL2ID.get(self.annotated_type if gold else self.db_type, IGNORE)
        side = PLACEMENTS.index(self.placement) if self.placement in PLACEMENTS else IGNORE
        return label, gold, side


@dataclass(frozen=True)
class Document:
    id: int
    library: str
    doc: str
    split: str
    volume: str
    periodical: str
    copy_group: str
    first: int
    length: int


@dataclass(frozen=True)
class Window:
    """Index row IDs in stream order. IDs are metadata, never model inputs."""
    rows: tuple[int, ...]
    source: str = 'fixed'
    scan_start: bool = False
    scan_end: bool = False
    document_metadata: bool = False


def join_stratum(a: Document, b: Document):
    if a.library == b.library and a.volume and a.volume == b.volume:
        return 'same_volume'
    if (a.library == b.library and a.periodical and a.periodical == b.periodical
            and a.volume and b.volume and a.volume != b.volume):
        return 'cross_volume'
    return 'unrestricted'
