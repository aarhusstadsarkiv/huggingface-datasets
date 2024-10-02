from re import compile as re_compile
from pathlib import Path
from typing import Generator
from typing import Optional
from typing import Sequence
from urllib.parse import unquote

from click import argument
from click import command
from click import option
from click import Path as ClickPath
from xmltodict import parse as parse_xml


transkribus_metadata_tag_pattern = re_compile(r"\s*<TranskribusMetadata[^>]*/?>([^<]*</TranskribusMetadata>)?")


def find_file(path: Path, name: str) -> Generator[Path, None, None]:
    if not path.is_dir():
        raise NotADirectoryError(path)

    items: list[Path] = [f for f in path.iterdir() if f.is_file() or f.is_dir()]
    file: Optional[Path] = next((f for f in items if f.is_file() and f.name == name), None)

    if file:
        yield file
    else:
        yield from (f for d in items if d.is_dir() for f in find_file(d, name))


def remove_transkribus_metadata(xml: str) -> str:
    return transkribus_metadata_tag_pattern.sub("", xml)


def dataset_generator_transkribus(root: Path, collections: Sequence[int]) -> Generator[dict[str, any], None, None]:
    for metadata_path in find_file(root.resolve(), "metadata.xml"):
        metadata_root: Path = metadata_path.parent
        metadata: dict = parse_xml(metadata_path.read_text("utf-8"), force_list=("colList",))

        doc_collections: list[int] = [int(c["colId"]) for c in metadata["trpDocMetadata"]["collectionList"]["colList"]]
        if collections and not set(collections).intersection(doc_collections):
            return

        doc_id: int = int(metadata["trpDocMetadata"]["docId"])
        title: str = metadata["trpDocMetadata"]["title"]

        mets: dict = parse_xml(metadata_root.joinpath("mets.xml").read_text("utf-8"))
        paths: dict[str, str] = {
            file["@ID"]: unquote(file["ns3:FLocat"]["@ns2:href"])
            for group in mets["ns3:mets"]["ns3:fileSec"]["ns3:fileGrp"]["ns3:fileGrp"]
            for file in group["ns3:file"]
        }

        for struct in mets["ns3:mets"]["ns3:structMap"]["ns3:div"]["ns3:div"]:
            img_id = next((i for f in struct["ns3:fptr"] if (i := f["ns3:area"]["@FILEID"]).startswith("IMG_")), None)
            alto_id = next((i for f in struct["ns3:fptr"] if (i := f["ns3:area"]["@FILEID"]).startswith("ALTO_")), None)
            page_id = next(
                (i for f in struct["ns3:fptr"] if (i := f["ns3:area"]["@FILEID"]).startswith("PAGEXML_")), None
            )
            yield {
                "image": str(metadata_root.joinpath(paths[img_id])),
                "doc_id": doc_id,
                "sequence": struct["@ORDER"],
                "alto": metadata_root.joinpath(paths[alto_id]).read_text("utf-8") if alto_id else "",
                "page": remove_transkribus_metadata(metadata_root.joinpath(paths[page_id]).read_text("utf-8")) if page_id else "",
            }


@command("transkribus-dataset")
@argument("repository", required=True)
@argument("folder", type=ClickPath(exists=True, file_okay=False, readable=True, resolve_path=True))
@option("--config-name", type=str, default="default")
@option("--collection", type=int, multiple=True)
@option("--token", type=str, envvar="HUGGINGFACE_TOKEN", default=None, show_envvar=True)
def app_transkribus(repository: str, folder: str, config_name: str, collection: tuple[int, ...], token: Optional[str]):
    print("Loading dataset... ", end="\r", flush=True)
    from datasets import Dataset
    from datasets.features import Features
    from datasets.features import Image
    from datasets.features import Value

    dataset: Dataset = Dataset.from_generator(
        lambda: dataset_generator_transkribus(Path(folder), collection),
        features=Features(
            {
                "image": Image(),
                "doc_id": Value("int64"),
                "sequence": Value("int16"),
                "alto": Value("string"),
                "page": Value("string"),
            }
        ),
        config_name=config_name,
    )
    print("Dataset loaded     ")

    print("Pushing dataset... ", end="\r", flush=True)
    dataset.push_to_hub(repository, config_name=config_name, token=token)
    print("Dataset Pushed     ")


if __name__ == "__main__":
    app_transkribus()
