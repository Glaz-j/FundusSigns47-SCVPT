# Recommended Data Format for Future FundusSigns Releases

The dataset should be distributed as a metadata-first package. Raw images can remain in a controlled-access object store, while labels and metadata are versioned as tabular files.

## Recommended Directory Layout

```text
FundusSigns-47/
  dataset_card.md
  labels/
    lesion_label_map.json
    diagnosis_label_map.json
  manifests/
    manifest.parquet
    manifest.csv
    manifest_public_schema.csv
  splits/
    split_v1_patient_hash.csv
  metadata/
    image_metadata.parquet
    annotation_metadata.parquet
  checksums/
    image_sha256.csv
```

## Core Manifest Columns

- `sample_id`: stable de-identified image-level identifier.
- `image_path`: relative path or object-store URI, never an absolute local path.
- `split`: `train`, `val`, `test`, or external split name.
- `lesion_ids`: pipe-separated 1-based sign IDs, for example `1|6|37`.
- `diagnosis_ids`: optional pipe-separated diagnosis IDs.
- `num_lesions`: number of positive sign labels.
- `image_width`, `image_height`: original image geometry.
- `laterality`: `OD`, `OS`, or unknown if available after de-identification.
- `source_site`: de-identified acquisition site or batch code.
- `quality_label`: optional image-quality flag.
- `sha256`: checksum of the image file.

## Columns to Avoid in a Public or Semi-Public Manifest

- Patient name, birthday, phone number, medical-record number, or free-text source filename with identifiers.
- Absolute source paths such as `/path/to/user/eyedataset/...`.
- Raw DICOM UID values unless governance explicitly permits them.
- Doctor IDs or annotation-file paths that identify staff or institutions.

## Preferred Storage Format

- Use `parquet` as the canonical machine-readable manifest because it preserves types and is compact.
- Export `csv` as a convenience view for simple scripts.
- Keep `lesion_label_map.json` as the authority for label IDs and names.
- Keep splits in a separate versioned file so future revisions can add images without changing historical test membership.

## Why This Helps

This structure lets collaborators run inference with only:

```text
manifest.csv + image folder/object-store access + lesion_label_map.json
```

It also keeps the private clinical dataset separable from the model/code release, which is the safer route for now.
