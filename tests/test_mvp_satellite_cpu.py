from __future__ import annotations

import io
import json
import shutil
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import rasterio
from botocore.exceptions import ClientError
from pydantic import SecretStr
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_contracts.mvp.contracts.common import TimeWindow
from fireviewer_orchestrator.mvp.gpu.sagemaker_service import GeoGpuResponse
from fireviewer_evidence_ingestion.mvp.satellite_cpu import (
    AzureFederatedSageMakerAsyncProvider,
    CanonicalPrithviRasterBuilder,
    PreparedSatelliteRaster,
    SageMakerAsyncConfig,
    SatelliteCpuError,
    SatelliteCpuWorker,
    build_prithvi_request,
)
from fireviewer_evidence_ingestion.mvp.satellite_observations import (
    CdseObservationS3Config,
    CdseS3ObservationAssetReader,
    ClmsRasterWindow,
    SatelliteAssetReceipt,
    SatelliteObservationAsset,
    SatelliteObservationCpuWorker,
    Sentinel1ChangeWindow,
    Sentinel2ChangeWindow,
    _frp_sample_count,
    _validate_clms_window_size,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendIncidentDaySatelliteArtifact,
    DurableEventEvidence,
)

_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
_DESCRIPTIONS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")


def test_clms_window_pixel_limit_is_enforced_before_raster_read() -> None:
    with pytest.raises(SatelliteCpuError, match="clms_satellite_window_too_large"):
        _validate_clms_window_size(width=2_001, height=2_000, maximum_pixels=4_000_000)


def test_sentinel3_sample_limit_is_enforced_before_netcdf_read() -> None:
    class OversizedVariable:
        shape = (500_001,)

    with pytest.raises(SatelliteCpuError, match="sentinel3_frp_sample_limit_exceeded"):
        _frp_sample_count(OversizedVariable())


def _write_sources(directory: Path) -> tuple[dict[str, Path], tuple[float, float, float, float]]:
    paths: dict[str, Path] = {}
    crs = "EPSG:32631"
    full_bounds = (500_000.0, 4_999_600.0, 500_400.0, 5_000_000.0)
    bbox = transform_bounds(crs, "EPSG:4326", *full_bounds, densify_pts=21)
    for index, band in enumerate(_BANDS, start=1):
        resolution = 10 if band in {"B02", "B03", "B04"} else 20
        size = 40 if resolution == 10 else 20
        path = directory / f"{band}.jp2"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype="uint16",
            crs=crs,
            transform=from_origin(500_000, 5_000_000, resolution, resolution),
            nodata=0,
        ) as dataset:
            dataset.write(np.full((size, size), index * 1_000, dtype=np.uint16), 1)
        paths[band] = path
    return paths, tuple(float(value) for value in bbox)


def _artifact(acquired_at: datetime) -> BackendIncidentDaySatelliteArtifact:
    return BackendIncidentDaySatelliteArtifact.model_validate(
        {
            "artifact_revision_id": "EAR-CDSE-20260824",
            "provider_key": "copernicus-cdse",
            "collection_key": "sentinel-2-l2a",
            "semantic_role": "raw_earth_observation",
            "external_product_id": "S2B_PRODUCT_20260824",
            "source_url": "https://catalogue.dataspace.copernicus.eu/product",
            "content_hash": "a" * 64,
            "acquisition_start_at": acquired_at.isoformat(),
            "native_crs": "EPSG:32631",
            "footprint_geojson": {
                "type": "Polygon",
                "coordinates": [[[2, 44], [3, 44], [3, 46], [2, 46], [2, 44]]],
            },
            "resolution_m": 20,
            "quality_flags": {"eo:cloud_cover": 12.5},
            "license": "Copernicus Data Space Ecosystem terms",
            "attribution": "Contains modified Copernicus Sentinel data 2026",
            "materialization_state": "materialized",
            "materialization_bundle_id": "SMB-CDSE-20260824",
            "materialization_manifest_sha256": "b" * 64,
            "prithvi_bands": [
                {
                    "canonical_band": band,
                    "asset_name": band,
                    "source_checksum": "1220" + (f"{index:x}" * 64)[:64],
                    "content_sha256": f"{index:x}" * 64,
                    "size_bytes": 1_024,
                    "media_type": "image/jp2",
                    "gsd_m": 20,
                    "proj_code": "EPSG:32631",
                    "proj_shape": [20, 20],
                    "proj_transform": [20, 0, 500000, 0, -20, 5000000],
                    "content_path": (
                        "/api/v1/internal/satellite-materializations/"
                        f"SMB-CDSE-20260824/bands/{band}/content"
                    ),
                }
                for index, band in enumerate(_BANDS, start=1)
            ],
        }
    )


class _Repository:
    def __init__(self, durable: DurableEventEvidence) -> None:
        self.durable = durable

    def read(self, event_id: str) -> DurableEventEvidence:
        assert event_id == self.durable.event.event_id
        return self.durable


class _Fetcher:
    ephemeral_directory: Path | None = None
    bbox: tuple[float, float, float, float] | None = None

    def fetch(self, *, artifact, directory: Path):
        assert artifact.artifact_revision_id == "EAR-CDSE-20260824"
        self.ephemeral_directory = directory
        paths, self.bbox = _write_sources(directory)
        return paths


class _Provider:
    model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
    model_revision = "a3f2c410e45b8ac7417976614528a872f024d831"

    def __init__(self, geometry: dict[str, Any]) -> None:
        self.geometry = geometry
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        item = request.worker_input.items[0]
        assert tuple(item.satellite.bands) == _DESCRIPTIONS
        assert request.payloads[0].content_sha256
        annotation_id = "SA-CDSE-20260824"
        return GeoGpuResponse.model_validate(
            {
                "schema": "fireviewer.geo-gpu-response.v1",
                "request_id": request.request_id,
                "operation": "prithvi.burned_area",
                "status": "completed",
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "result": {
                    "annotations": {
                        item.input_id: [
                            {
                                "annotation_id": annotation_id,
                                "evidence_id": item.input_id,
                                "evidence_kind": "satellite_image",
                                "semantic_anchor": "burned_area_polygon",
                                "source_geometry_normalized": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.1]]
                                    ],
                                },
                                "model_score": 0.88,
                            }
                        ]
                    },
                    "spatial_proposals": {
                        item.input_id: [
                            {
                                "proposal_id": "SP-CDSE-20260824",
                                "annotation_id": annotation_id,
                                "status": "projected_geometry",
                                "proposal_kind": "burned_area_polygon",
                                "observed_at": item.satellite.acquired_at.isoformat(),
                                "geometry_origin": "SATELLITE_GEOTRANSFORM",
                                "geometry_geojson": self.geometry,
                                "horizontal_accuracy_m": item.satellite.resolution_m,
                                "reference_bundle_sha256": (
                                    request.worker_input.reference_bundle.manifest_sha256
                                ),
                                "uncertainty_codes": ["burned_area_model_proposal"],
                            }
                        ]
                    },
                },
                "reason_codes": [],
            }
        )


class _Publisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def publish(self, *, candidate_id: str, payload):
        assert candidate_id == payload["analysis_id"]
        self.payloads.append(dict(payload))
        return None


class _ObservationPublisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def publish(self, *, candidate_id: str, payload):
        assert candidate_id == payload["analysis_id"]
        self.payloads.append(dict(payload))
        return None


class _LocalObservationReader:
    def __init__(
        self,
        *,
        clms_paths: dict[str, Path] | None = None,
        frp_path: Path | None = None,
        sentinel1_window: Sentinel1ChangeWindow | None = None,
        sentinel2_window: Sentinel2ChangeWindow | None = None,
    ) -> None:
        self.clms_paths = clms_paths
        self.frp_path = frp_path
        self.sentinel1_window = sentinel1_window
        self.sentinel2_window = sentinel2_window
        self.frp_ephemeral_path: Path | None = None

    def read_clms_window(self, *, assets, bbox):
        assert self.clms_paths is not None
        arrays = []
        masks = []
        receipts = []
        transform = None
        for asset in assets:
            path = self.clms_paths[asset.asset_name]
            with rasterio.open(path) as dataset:
                masked = dataset.read(1, masked=True)
                transform = dataset.transform
            arrays.append(
                np.asarray(masked.filled(0), dtype=np.float64) * float(asset.raster_scale)
            )
            masks.append(~np.ma.getmaskarray(masked))
            content = path.read_bytes()
            receipts.append(
                SatelliteAssetReceipt(
                    asset_name=asset.asset_name,
                    source_checksum=asset.file_checksum,
                    derived_content_sha256=sha256(content).hexdigest(),
                    bytes_read=len(content),
                )
            )
        assert transform is not None
        return ClmsRasterWindow(
            day_of_burn=arrays[0],
            burn_probability=arrays[1],
            burn_fraction=arrays[2],
            valid_masks=tuple(masks),
            transform=transform,
            receipts=tuple(receipts),
        )

    def fetch_frp_file(self, *, asset, output_path: Path):
        assert self.frp_path is not None
        content = self.frp_path.read_bytes()
        assert len(content) == asset.file_size_bytes
        shutil.copyfile(self.frp_path, output_path)
        self.frp_ephemeral_path = output_path
        return SatelliteAssetReceipt(
            asset_name=asset.asset_name,
            source_checksum=asset.file_checksum,
            derived_content_sha256=sha256(content).hexdigest(),
            bytes_read=len(content),
        )

    def read_sentinel2_change_window(self, *, reference_assets, observation_assets, bbox):
        assert tuple(item.asset_name for item in reference_assets) == (
            "B04_20m",
            "B8A_20m",
            "B11_20m",
            "B12_20m",
            "SCL_20m",
        )
        assert tuple(item.asset_name for item in observation_assets) == tuple(
            item.asset_name for item in reference_assets
        )
        assert self.sentinel2_window is not None
        return self.sentinel2_window

    def read_sentinel1_change_window(self, *, reference_artifact, observation_artifact, bbox):
        assert reference_artifact.quality_flags["temporal_role"] == "pre_fire_reference"
        assert observation_artifact.quality_flags["temporal_role"] == "post_fire_observation"
        assert self.sentinel1_window is not None
        return self.sentinel1_window


def test_builder_aligns_six_bands_on_b11_grid(tmp_path: Path) -> None:
    paths, bbox = _write_sources(tmp_path)
    output = CanonicalPrithviRasterBuilder().build(
        band_paths=paths,
        incident_bbox=bbox,
        output_path=tmp_path / "canonical.tif",
    )

    with rasterio.open(output.path) as dataset:
        assert dataset.count == 6
        assert dataset.descriptions == _DESCRIPTIONS
        assert dataset.res == (20, 20)
        assert dataset.width == 20
        assert dataset.height == 20
        assert dataset.read(1).mean() == 1_000
        assert dataset.read(6).mean() == 6_000
    assert output.sha256
    assert output.size_bytes == output.path.stat().st_size


def test_worker_uses_ephemeral_bands_and_publishes_only_derived_geometry() -> None:
    acquired_at = datetime(2026, 8, 24, 10, tzinfo=UTC)
    probe_directory = Path("unused")
    _paths, bbox = _write_sources_for_bbox_only(probe_directory, acquired_at)
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[1]],
            ]
        ],
    }
    durable = DurableEventEvidence(
        event=EventEvidenceV1(
            event_id="AN-CDSE-20260824",
            time_window=TimeWindow(
                from_at=datetime(2026, 8, 23, 22, tzinfo=UTC),
                to_at=datetime(2026, 8, 24, 22, tzinfo=UTC),
            ),
        ),
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="c" * 64,
        incident_id="FR-26-00001",
        research_target_kind="incident_day",
        satellite_artifact_tickets=(_artifact(acquired_at),),
        incident_day_episode_id="EP-CDSE-20260824",
        incident_day_local_date=date(2026, 8, 24),
        incident_day_timezone="Europe/Paris",
        incident_day_bbox=bbox,
    )
    fetcher = _Fetcher()
    provider = _Provider(geometry)
    publisher = _Publisher()

    receipt = SatelliteCpuWorker(
        repository=_Repository(durable),
        band_fetcher=fetcher,
        raster_builder=CanonicalPrithviRasterBuilder(),
        provider=provider,
        publisher=publisher,
    ).run(durable.event.event_id)

    assert receipt.processed == 1
    assert receipt.statuses == ("completed",)
    assert len(provider.requests) == 1
    assert len(publisher.payloads) == 1
    persisted = publisher.payloads[0]
    assert persisted["spatial_proposals"][0]["geometry_geojson"] == geometry
    assert "content_base64" not in str(persisted)
    assert persisted["prithvi_input_sha256"]
    assert fetcher.ephemeral_directory is not None
    assert not fetcher.ephemeral_directory.exists()


def _write_sources_for_bbox_only(
    _directory: Path, _acquired_at: datetime
) -> tuple[dict[str, Path], tuple[float, float, float, float]]:
    full_bounds = (500_000.0, 4_999_600.0, 500_400.0, 5_000_000.0)
    bbox = transform_bounds("EPSG:32631", "EPSG:4326", *full_bounds, densify_pts=21)
    return {}, tuple(float(value) for value in bbox)


def _client_error(status: int, operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": str(status), "Message": "test"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


def test_sagemaker_async_poll_resumes_without_a_second_paid_invocation(
    tmp_path: Path,
) -> None:
    acquired_at = datetime(2026, 8, 24, 10, tzinfo=UTC)
    raster_path = tmp_path / "canonical.tif"
    raster_path.write_bytes(b"canonical-six-band-raster")
    raster = PreparedSatelliteRaster(
        path=raster_path,
        sha256=sha256(raster_path.read_bytes()).hexdigest(),
        size_bytes=raster_path.stat().st_size,
        crs="EPSG:32631",
        width=20,
        height=20,
        geotransform=(500_000, 20, 0, 5_000_000, 0, -20),
        bbox_wgs84=(2.9, 45.0, 3.0, 45.1),
        resolution_m=20,
    )
    durable = DurableEventEvidence(
        event=EventEvidenceV1(
            event_id="AN-CDSE-RESUME",
            time_window=TimeWindow(
                from_at=datetime(2026, 8, 23, 22, tzinfo=UTC),
                to_at=datetime(2026, 8, 24, 22, tzinfo=UTC),
            ),
        ),
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="c" * 64,
        incident_id="FR-26-00001",
        research_target_kind="incident_day",
        satellite_artifact_tickets=(_artifact(acquired_at),),
        incident_day_episode_id="EP-CDSE-RESUME",
        incident_day_local_date=date(2026, 8, 24),
        incident_day_timezone="Europe/Paris",
        incident_day_bbox=raster.bbox_wgs84,
    )
    request = build_prithvi_request(
        durable=durable,
        artifact=durable.satellite_artifact_tickets[0],
        raster=raster,
        request_id="GEO-CDSE-RESUME",
    )
    expected = _Provider(
        {
            "type": "Polygon",
            "coordinates": [[[2.9, 45.0], [3.0, 45.0], [3.0, 45.1], [2.9, 45.0]]],
        }
    ).invoke(request)

    class S3:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.output_ready = False

        def put_object(self, *, Bucket, Key, Body, **kwargs):
            assert Bucket == "fireviewer-geo-ai-123456789012-eu-west-3"
            if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
                raise _client_error(412, "PutObject")
            if kwargs.get("IfMatch") is not None and Key not in self.objects:
                raise _client_error(412, "PutObject")
            self.objects[Key] = bytes(Body)
            return {"ETag": '"etag-1"'}

        def get_object(self, *, Bucket, Key):
            assert Bucket == "fireviewer-geo-ai-123456789012-eu-west-3"
            if Key == "async/output/result.json" and self.output_ready:
                body = json.dumps(
                    expected.model_dump(mode="json", by_alias=True),
                    separators=(",", ":"),
                ).encode()
                return {"Body": io.BytesIO(body)}
            if Key not in self.objects:
                raise _client_error(404, "GetObject")
            return {"Body": io.BytesIO(self.objects[Key])}

        def head_object(self, *, Bucket, Key):
            assert Bucket == "fireviewer-geo-ai-123456789012-eu-west-3"
            if Key == "async/output/result.json" and self.output_ready:
                return {"ContentLength": 1}
            raise _client_error(404, "HeadObject")

    class Runtime:
        def __init__(self) -> None:
            self.invocations = 0

        def invoke_endpoint_async(self, **kwargs):
            self.invocations += 1
            assert kwargs["InferenceId"] == request.request_id
            return {
                "OutputLocation": (
                    "s3://fireviewer-geo-ai-123456789012-eu-west-3/async/output/result.json"
                )
            }

    now = [datetime(2026, 8, 24, 12, tzinfo=UTC)]
    s3 = S3()
    runtime = Runtime()
    provider = AzureFederatedSageMakerAsyncProvider(
        SageMakerAsyncConfig(
            region_name="eu-west-3",
            role_arn="arn:aws:iam::123456789012:role/fireviewer-geo",
            endpoint_name="fireviewer-geo-async-0123456789abcdef",
            bucket_name="fireviewer-geo-ai-123456789012-eu-west-3",
            maximum_wait_seconds=30,
            poll_seconds=10,
        ),
        web_token_provider=lambda: "unused",
        sts_client=object(),
        clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + timedelta(seconds=seconds)),
    )
    provider._s3 = s3
    provider._runtime = runtime
    provider._expires_at = now[0] + timedelta(hours=1)

    with pytest.raises(SatelliteCpuError, match="sagemaker_async_timeout") as first:
        provider.invoke(request)
    assert first.value.retryable is True
    assert runtime.invocations == 1

    s3.output_ready = True
    resumed = provider.invoke(request)

    assert resumed.request_id == request.request_id
    assert runtime.invocations == 1


def _observation_artifact(
    *,
    collection_key: str,
    processor: str,
    assets: list[dict[str, Any]],
    acquired_at: datetime,
    resolution_m: float,
    artifact_revision_id: str | None = None,
    temporal_role: str | None = None,
) -> BackendIncidentDaySatelliteArtifact:
    return BackendIncidentDaySatelliteArtifact.model_validate(
        {
            "artifact_revision_id": artifact_revision_id
            or f"EAR-{sha256(collection_key.encode()).hexdigest()[:24]}",
            "provider_key": "copernicus-cdse",
            "collection_key": collection_key,
            "semantic_role": "interpreted_observation",
            "external_product_id": f"PRODUCT-{processor}",
            "source_url": "https://stac.dataspace.copernicus.eu/v1/item",
            "content_hash": "a" * 64,
            "acquisition_start_at": acquired_at.isoformat(),
            "native_crs": "EPSG:4326",
            "footprint_geojson": {
                "type": "Polygon",
                "coordinates": [[[-180, -60], [180, -60], [180, 80], [-180, 80], [-180, -60]]],
            },
            "resolution_m": resolution_m,
            "quality_flags": {
                "satellite_observation_processor": processor,
                "satellite_observation_assets": assets,
                **({"temporal_role": temporal_role} if temporal_role is not None else {}),
            },
            "license": "Copernicus data policy",
            "attribution": "European Union Copernicus programme",
            "materialization_state": "not_required",
        }
    )


def _observation_durable(
    artifact: BackendIncidentDaySatelliteArtifact,
    *additional_artifacts: BackendIncidentDaySatelliteArtifact,
) -> DurableEventEvidence:
    return DurableEventEvidence(
        event=EventEvidenceV1(
            event_id="AN-DIE-20260709",
            time_window=TimeWindow(
                from_at=datetime(2026, 7, 8, 22, tzinfo=UTC),
                to_at=datetime(2026, 7, 9, 22, tzinfo=UTC),
            ),
        ),
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="b" * 64,
        incident_id="FR-26-00001",
        research_target_kind="incident_day",
        satellite_artifact_tickets=(artifact, *additional_artifacts),
        incident_day_episode_id="E01",
        incident_day_local_date=date(2026, 7, 9),
        incident_day_timezone="Europe/Paris",
        incident_day_bbox=(5.36, 44.74, 5.38, 44.76),
    )


def test_clms_daily_cogs_produce_a_clipped_burn_scar_without_raw_rasters(
    tmp_path: Path,
) -> None:
    target_day = date(2026, 7, 9).timetuple().tm_yday
    transform = from_origin(5.36, 44.76, 0.005, 0.005)
    raw_arrays = {
        "ba300_dob_nrt": np.array(
            [[0, 0, 0, 0], [0, target_day, target_day, 0], [0, target_day, 0, 0], [0, 0, 0, 0]],
            dtype=np.int16,
        ),
        "ba300_cp_nrt": np.array(
            [[0, 0, 0, 0], [0, 900, 800, 0], [0, 700, 0, 0], [0, 0, 0, 0]],
            dtype=np.int16,
        ),
        "ba300_bf_nrt": np.array(
            [[0, 0, 0, 0], [0, 600, 500, 0], [0, 400, 0, 0], [0, 0, 0, 0]],
            dtype=np.int16,
        ),
    }
    paths: dict[str, Path] = {}
    asset_payloads: list[dict[str, Any]] = []
    for asset_name, values in raw_arrays.items():
        path = tmp_path / f"{asset_name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="int16",
            crs="EPSG:4326",
            transform=transform,
            nodata=-1,
        ) as dataset:
            dataset.write(values, 1)
        paths[asset_name] = path
        asset_payloads.append(
            {
                "asset_name": asset_name,
                "object_uri": f"s3://eodata/CLMS/test/{asset_name}.tif",
                "media_type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file_size_bytes": path.stat().st_size,
                "file_checksum": sha256(path.read_bytes()).hexdigest(),
                "proj_code": "EPSG:4326",
                "proj_shape": [4, 4],
                "proj_transform": list(transform.to_gdal()),
                "nodata": -1,
                "data_type": "int16",
                "raster_scale": 1 if asset_name == "ba300_dob_nrt" else 0.001,
            }
        )
    burned_area_collection = "clms_ba_global_300m_daily_v4_cog"
    artifact = _observation_artifact(
        collection_key=burned_area_collection,
        processor="clms_burned_area_daily_v1",
        assets=asset_payloads,
        acquired_at=datetime(2026, 7, 9, tzinfo=UTC),
        resolution_m=300,
    )
    durable = _observation_durable(artifact)
    publisher = _ObservationPublisher()

    receipt = SatelliteObservationCpuWorker(
        repository=_Repository(durable),
        asset_reader=_LocalObservationReader(clms_paths=paths),
        publisher=publisher,
    ).run(durable.event.event_id, artifact.artifact_revision_id)

    assert receipt.status == "completed"
    assert len(publisher.payloads) == 1
    payload = publisher.payloads[0]
    assert payload["raw_satellite_content_stored"] is False
    assert len(payload["asset_receipts"]) == 3
    assert len(payload["observations"]) == 3
    observations = payload["observations"]
    assert all(
        observation["geometry_geojson"]["type"] in {"Polygon", "MultiPolygon"}
        for observation in observations
    )
    assert {observation["metrics"]["pixel_count"] for observation in observations} == {3}
    assert (
        sum(
            observation["metrics"]["probability_bucket_pixel_count"] for observation in observations
        )
        == 3
    )
    assert {round(observation["confidence"], 2) for observation in observations} == {0.7, 0.8, 0.9}
    assert {observation["metrics"]["target_day_of_year"] for observation in observations} == {
        target_day
    }
    assert "object_uri" not in str(payload)


def test_sentinel1_prefire_postfire_vvvh_change_is_low_confidence_second_opinion() -> None:
    asset_payload = {
        "asset_name": "openeo_vv_vh",
        "object_uri": "s3://eodata/openeo-sentinel1/test/source-item.json",
        "media_type": "application/geo+json",
        "file_size_bytes": 1_024,
        "file_checksum": "a" * 64,
    }
    reference = _observation_artifact(
        collection_key="sentinel-1-grd",
        processor="sentinel1_vvvh_change_v1",
        assets=[asset_payload],
        acquired_at=datetime(2026, 7, 1, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S1-PREFIRE-DIE-20260701",
        temporal_role="pre_fire_reference",
    )
    observation = _observation_artifact(
        collection_key="sentinel-1-grd",
        processor="sentinel1_vvvh_change_v1",
        assets=[asset_payload],
        acquired_at=datetime(2026, 7, 9, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S1-POSTFIRE-DIE-20260709",
        temporal_role="post_fire_observation",
    )
    pre = {
        "VV": np.full((4, 4), 0.1, dtype=np.float32),
        "VH": np.full((4, 4), 0.04, dtype=np.float32),
    }
    post = {key: value.copy() for key, value in pre.items()}
    post["VV"][1:3, 1:3] = 0.01
    post["VH"][1:3, 1:3] = 0.004
    transform = from_origin(5.36, 44.76, 0.005, 0.005)

    def receipt(seed: str) -> SatelliteAssetReceipt:
        return SatelliteAssetReceipt(
            asset_name="openeo_vv_vh",
            source_checksum="a" * 64,
            derived_content_sha256=sha256(seed.encode()).hexdigest(),
            bytes_read=1_024,
        )

    window = Sentinel1ChangeWindow(
        pre=pre,
        post=post,
        transform=transform,
        crs="EPSG:4326",
        receipts_pre=(receipt("pre"),),
        receipts_post=(receipt("post"),),
    )
    durable = _observation_durable(observation, reference)
    publisher = _ObservationPublisher()

    result = SatelliteObservationCpuWorker(
        repository=_Repository(durable),
        asset_reader=_LocalObservationReader(sentinel1_window=window),
        publisher=publisher,
        sentinel1_vv_change_threshold_db=1.5,
        sentinel1_vh_change_threshold_db=1.5,
        openeo_maximum_authorized_credits=1,
    ).run(durable.event.event_id, observation.artifact_revision_id)

    assert result.status == "completed"
    payload = publisher.payloads[0]
    assert payload["processor"] == "sentinel1_vvvh_change_v1"
    assert payload["reference_artifact_revision_id"] == reference.artifact_revision_id
    assert len(payload["asset_receipts"]) == 2
    assert payload["coverage_metrics"] == {
        "valid_pixel_count": 16,
        "invalid_fraction": 0.0,
        "changed_pixel_count": 4,
    }
    observation_payload = payload["observations"][0]
    assert observation_payload["metrics"]["changed_pixel_count"] == 4
    assert observation_payload["confidence"] <= 0.45
    assert observation_payload["geometry_geojson"]["type"] in {"Polygon", "MultiPolygon"}


def test_sentinel1_without_openeo_authorization_is_explicitly_unavailable() -> None:
    asset_payload = {
        "asset_name": "openeo_vv_vh",
        "object_uri": "s3://eodata/openeo-sentinel1/test/source-item.json",
        "media_type": "application/geo+json",
        "file_size_bytes": 1_024,
        "file_checksum": "a" * 64,
    }
    reference = _observation_artifact(
        collection_key="sentinel-1-grd",
        processor="sentinel1_vvvh_change_v1",
        assets=[asset_payload],
        acquired_at=datetime(2026, 7, 1, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S1-PREFIRE-NOT-AUTHORIZED",
        temporal_role="pre_fire_reference",
    )
    observation = _observation_artifact(
        collection_key="sentinel-1-grd",
        processor="sentinel1_vvvh_change_v1",
        assets=[asset_payload],
        acquired_at=datetime(2026, 7, 9, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S1-POSTFIRE-NOT-AUTHORIZED",
        temporal_role="post_fire_observation",
    )
    durable = _observation_durable(observation, reference)
    publisher = _ObservationPublisher()

    result = SatelliteObservationCpuWorker(
        repository=_Repository(durable),
        asset_reader=_LocalObservationReader(),
        publisher=publisher,
    ).run(durable.event.event_id, observation.artifact_revision_id)

    assert result.status == "unavailable"
    assert result.processed == 1
    payload = publisher.payloads[0]
    assert payload["unavailable_reason"] == "cdse_openeo_not_authorized"
    assert payload["observations"] == []
    assert payload["asset_receipts"] == []
    assert payload["processing_parameters"] == {}
    assert payload["raw_satellite_content_stored"] is False


def test_openeo_sentinel1_reader_is_bounded_and_never_exposes_its_token() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openeosh.dataspace.copernicus.eu/1.2/result"
        assert request.headers["authorization"] == "Bearer token-" + ("t" * 40)
        payload = json.loads(request.content)
        captured.append(payload)
        load = payload["process"]["process_graph"]["load"]
        extent = load["arguments"]["spatial_extent"]
        assert load["arguments"]["bands"] == ["VV", "VH"]
        assert (
            payload["process"]["process_graph"]["backscatter"]["arguments"]["coefficient"]
            == "sigma0-ellipsoid"
        )
        assert "token-" not in json.dumps(payload)
        memory = MemoryFile()
        with memory.open(
            driver="GTiff",
            width=extent["width"],
            height=extent["height"],
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(5.36, 44.76, 0.0001, 0.0001),
        ) as dataset:
            dataset.write(
                np.full((extent["height"], extent["width"]), 0.1, dtype=np.float32),
                1,
            )
            dataset.write(
                np.full((extent["height"], extent["width"]), 0.04, dtype=np.float32),
                2,
            )
        content = memory.read()
        memory.close()
        return httpx.Response(200, content=content, headers={"Content-Type": "image/tiff"})

    reference = _observation_artifact(
        collection_key="sentinel-1-grd",
        processor="sentinel1_vvvh_change_v1",
        assets=[],
        acquired_at=datetime(2026, 7, 1, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S1-OPENEO-PRE",
        temporal_role="pre_fire_reference",
    )
    observation = _observation_artifact(
        collection_key="sentinel-1-grd",
        processor="sentinel1_vvvh_change_v1",
        assets=[],
        acquired_at=datetime(2026, 7, 9, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S1-OPENEO-POST",
        temporal_role="post_fire_observation",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reader = CdseS3ObservationAssetReader(
            CdseObservationS3Config(
                access_key=SecretStr("access-key"),
                secret_key=SecretStr("secret-key-" + ("s" * 24)),
                openeo_invocation_enabled=True,
                openeo_access_token=SecretStr("token-" + ("t" * 40)),
                openeo_maximum_authorized_credits=1,
            ),
            s3_client=object(),
            http_client=client,
        )
        result = reader.read_sentinel1_change_window(
            reference_artifact=reference,
            observation_artifact=observation,
            bbox=(5.36, 44.74, 5.3604, 44.7404),
        )

    assert len(captured) == 2
    assert result.pre["VV"].shape == result.post["VV"].shape
    assert result.receipts_pre[0].source_checksum == reference.content_hash
    assert result.receipts_post[0].source_checksum == observation.content_hash


def test_sentinel2_reader_materializes_bounded_assets_before_reading() -> None:
    transform = from_origin(5.36, 44.76, 0.005, 0.005)
    objects: dict[str, bytes] = {}

    def raster_bytes(value: int) -> bytes:
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                width=4,
                height=4,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(np.full((4, 4), value, dtype=np.uint16), 1)
            return memory.read()

    def assets(temporal_role: str, base_value: int) -> tuple[SatelliteObservationAsset, ...]:
        payloads = []
        for index, asset_name in enumerate(
            ("B04_20m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"),
            start=1,
        ):
            key = (
                "Sentinel-2/MSI/L2A/2026/07/09/PRODUCT.SAFE/GRANULE/TEST/IMG_DATA/"
                f"R20m/{temporal_role}-{asset_name}.jp2"
            )
            content = raster_bytes(4 if asset_name == "SCL_20m" else base_value + index)
            objects[key] = content
            payloads.append(
                SatelliteObservationAsset.model_validate(
                    {
                        "asset_name": asset_name,
                        "object_uri": f"s3://eodata/{key}",
                        "media_type": "image/jp2",
                        "file_size_bytes": len(content),
                        "file_checksum": sha256(content).hexdigest(),
                        "proj_code": "EPSG:4326",
                        "proj_shape": [4, 4],
                        "proj_transform": list(transform.to_gdal()),
                        "nodata": 0,
                        "data_type": "uint16",
                        "raster_scale": 1,
                    }
                )
            )
        return tuple(payloads)

    reference_assets = assets("pre", 1_000)
    observation_assets = assets("post", 2_000)

    class S3:
        def __init__(self) -> None:
            self.requested: list[str] = []

        def get_object(self, *, Bucket, Key):
            assert Bucket == "eodata"
            self.requested.append(Key)
            return {"Body": io.BytesIO(objects[Key])}

    s3 = S3()
    reader = CdseS3ObservationAssetReader(
        CdseObservationS3Config(
            access_key=SecretStr("access-key"),
            secret_key=SecretStr("secret-key-" + ("s" * 24)),
        ),
        s3_client=s3,
    )
    try:
        result = reader.read_sentinel2_change_window(
            reference_assets=reference_assets,
            observation_assets=observation_assets,
            bbox=(5.36, 44.74, 5.38, 44.76),
        )
    finally:
        reader.close()

    assert len(s3.requested) == 10
    assert set(s3.requested) == set(objects)
    assert all(np.isfinite(array).all() for array in (*result.pre.values(), *result.post.values()))
    assert result.pre["B8A_20m"].mean() == pytest.approx(1_002)
    assert result.post["B8A_20m"].mean() == pytest.approx(2_002)
    assert len(result.receipts_pre) == 5
    assert len(result.receipts_post) == 5


def test_sentinel2_reader_rejects_download_above_ephemeral_limit() -> None:
    transform = from_origin(5.36, 44.76, 0.005, 0.005)
    assets = tuple(
        SatelliteObservationAsset.model_validate(
            {
                "asset_name": asset_name,
                "object_uri": (
                    "s3://eodata/Sentinel-2/MSI/L2A/2026/07/09/PRODUCT.SAFE/"
                    f"GRANULE/TEST/IMG_DATA/R20m/{asset_name}.jp2"
                ),
                "media_type": "image/jp2",
                "file_size_bytes": 1_024,
                "file_checksum": f"{index:x}" * 64,
                "proj_code": "EPSG:4326",
                "proj_shape": [4, 4],
                "proj_transform": list(transform.to_gdal()),
                "nodata": 0,
                "data_type": "uint16",
                "raster_scale": 1,
            }
        )
        for index, asset_name in enumerate(
            ("B04_20m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"),
            start=1,
        )
    )
    reader = CdseS3ObservationAssetReader(
        CdseObservationS3Config(
            access_key=SecretStr("access-key"),
            secret_key=SecretStr("secret-key-" + ("s" * 24)),
            sentinel2_maximum_download_bytes=9 * 1_024,
        ),
        s3_client=object(),
    )
    try:
        with pytest.raises(SatelliteCpuError, match="sentinel2_download_too_large"):
            reader.read_sentinel2_change_window(
                reference_assets=assets,
                observation_assets=assets,
                bbox=(5.36, 44.74, 5.38, 44.76),
            )
    finally:
        reader.close()


def test_sentinel2_prefire_postfire_nbr_change_produces_burned_probability() -> None:
    asset_names = ("B04_20m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m")
    transform = from_origin(5.36, 44.76, 0.005, 0.005)
    assets = [
        {
            "asset_name": asset_name,
            "object_uri": (
                "s3://eodata/Sentinel-2/MSI/L2A/2026/07/09/PRODUCT.SAFE/"
                f"GRANULE/TEST/IMG_DATA/R20m/TEST_{asset_name}.jp2"
            ),
            "media_type": "image/jp2",
            "file_size_bytes": 32_768,
            "file_checksum": f"{index:x}" * 64,
            "proj_code": "EPSG:4326",
            "proj_shape": [4, 4],
            "proj_transform": list(transform.to_gdal()),
            "nodata": 0,
            "data_type": "uint16",
            "raster_scale": 1,
        }
        for index, asset_name in enumerate(asset_names, start=1)
    ]
    reference = _observation_artifact(
        collection_key="sentinel-2-l2a",
        processor="sentinel2_nbr_change_v1",
        assets=assets,
        acquired_at=datetime(2026, 7, 1, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S2-PREFIRE-DIE-20260701",
        temporal_role="pre_fire_reference",
    )
    observation = _observation_artifact(
        collection_key="sentinel-2-l2a",
        processor="sentinel2_nbr_change_v1",
        assets=assets,
        acquired_at=datetime(2026, 7, 9, 10, tzinfo=UTC),
        resolution_m=20,
        artifact_revision_id="EAR-S2-POSTFIRE-DIE-20260709",
        temporal_role="post_fire_observation",
    )
    pre = {
        "B04_20m": np.full((4, 4), 2_000, dtype=np.float32),
        "B8A_20m": np.full((4, 4), 8_000, dtype=np.float32),
        "B11_20m": np.full((4, 4), 2_500, dtype=np.float32),
        "B12_20m": np.full((4, 4), 2_000, dtype=np.float32),
        "SCL_20m": np.full((4, 4), 4, dtype=np.float32),
    }
    post = {key: value.copy() for key, value in pre.items()}
    post["B8A_20m"][1:3, 1:3] = 2_000
    post["B12_20m"][1:3, 1:3] = 6_000

    def receipts(seed: str) -> tuple[SatelliteAssetReceipt, ...]:
        return tuple(
            SatelliteAssetReceipt(
                asset_name=asset_name,
                source_checksum=f"{index:x}" * 64,
                derived_content_sha256=sha256(f"{seed}:{asset_name}".encode()).hexdigest(),
                bytes_read=64,
            )
            for index, asset_name in enumerate(asset_names, start=1)
        )

    raster_window = Sentinel2ChangeWindow(
        pre=pre,
        post=post,
        transform=transform,
        crs="EPSG:4326",
        receipts_pre=receipts("pre"),
        receipts_post=receipts("post"),
    )
    durable = _observation_durable(observation, reference)
    publisher = _ObservationPublisher()
    receipt = SatelliteObservationCpuWorker(
        repository=_Repository(durable),
        asset_reader=_LocalObservationReader(sentinel2_window=raster_window),
        publisher=publisher,
    ).run(durable.event.event_id, observation.artifact_revision_id)

    assert receipt.status == "completed"
    payload = publisher.payloads[0]
    assert payload["processor"] == "sentinel2_nbr_change_v1"
    assert payload["processor_revision"] == "fireviewer-sentinel2-nbr-change-cpu-1.3.0"
    assert len(payload["processing_context_sha256"]) == 64
    assert payload["result_id"].startswith("SATOBS-")
    assert payload["reference_artifact_revision_id"] == reference.artifact_revision_id
    assert len(payload["asset_receipts"]) == 10
    assert {item["source_artifact_revision_id"] for item in payload["asset_receipts"]} == {
        reference.artifact_revision_id,
        observation.artifact_revision_id,
    }
    result = payload["observations"][0]
    assert result["metrics"]["burned_pixel_count"] == 4
    assert result["metrics"]["valid_pixel_count"] == 16
    assert payload["valid_coverage_geojson"]["type"] == "Polygon"
    assert payload["coverage_metrics"] == {
        "valid_pixel_count": 16,
        "valid_fraction": 1.0,
        "cloud_fraction": 0.0,
        "no_data_fraction": 0.0,
        "scl_invalid_fraction": 0.0,
        "spectral_invalid_pixel_count": 0,
        "burned_pixel_count": 4,
        "water_pixel_count": 0,
        "water_positive_excluded_count": 0,
    }
    assert result["coverage_geojson"]["type"] == "Polygon"
    assert result["geometry_geojson"]["type"] in {"Polygon", "MultiPolygon"}

    no_change_publisher = _ObservationPublisher()
    no_change_window = Sentinel2ChangeWindow(
        pre=pre,
        post={key: value.copy() for key, value in pre.items()},
        transform=transform,
        crs="EPSG:4326",
        receipts_pre=receipts("no-change-pre"),
        receipts_post=receipts("no-change-post"),
    )
    no_change = SatelliteObservationCpuWorker(
        repository=_Repository(durable),
        asset_reader=_LocalObservationReader(sentinel2_window=no_change_window),
        publisher=no_change_publisher,
    ).run(durable.event.event_id, observation.artifact_revision_id)
    assert no_change.status == "no_observation"
    no_change_payload = no_change_publisher.payloads[0]
    assert no_change_payload["result_id"] == payload["result_id"]
    assert no_change_payload["processing_context_sha256"] == payload["processing_context_sha256"]
    assert no_change_payload["observations"] == []
    assert no_change_payload["valid_coverage_geojson"]["type"] == "Polygon"
    assert no_change_payload["coverage_metrics"]["valid_pixel_count"] == 16
    assert no_change_payload["coverage_metrics"]["burned_pixel_count"] == 0


@pytest.mark.parametrize(
    ("collection_key", "asset_name", "confidence_variable"),
    [
        ("sentinel-3-sl-2-frp-ntc", "FRP_in", "confidence"),
        (
            "sentinel-3-sl-2-frp-nrt",
            "FRP_MWIR1km_STANDARD",
            "confidence_level",
        ),
    ],
)
def test_sentinel3_frp_netcdf_produces_only_in_bbox_confident_hotspots(
    tmp_path: Path,
    collection_key: str,
    asset_name: str,
    confidence_variable: str,
) -> None:
    import h5py

    source_path = tmp_path / f"source-{asset_name}.nc"
    time_origin = datetime(2000, 1, 1, tzinfo=UTC)
    observation_times = [
        datetime(2026, 7, 9, 10, tzinfo=UTC),
        datetime(2026, 7, 9, 10, 1, tzinfo=UTC),
        datetime(2026, 7, 9, 10, 2, tzinfo=UTC),
    ]
    with h5py.File(source_path, "w") as dataset:
        dataset.create_dataset("latitude", data=[44.75, 44.751, 45.2], dtype="f8")
        dataset.create_dataset("longitude", data=[5.37, 5.371, 6.0], dtype="f8")
        dataset.create_dataset("FRP_MWIR", data=[18.5, 9.0, 80.0], dtype="f8")
        dataset.create_dataset("FRP_uncertainty_MWIR", data=[1.2, 2.0, 1.0], dtype="f8")
        dataset.create_dataset("IFOV_area", data=[1_000_000] * 3, dtype="f8")
        dataset.create_dataset(confidence_variable, data=[90, 20, 95], dtype="i2")
        dataset.create_dataset("classification", data=[1, 1, 2], dtype="i2")
        time_variable = dataset.create_dataset(
            "time",
            data=[
                int((value - time_origin).total_seconds() * 1_000_000)
                for value in observation_times
            ],
            dtype="i8",
        )
        time_variable.attrs["units"] = "microseconds since 2000-01-01 00:00:00 UTC"
    asset_payload = {
        "asset_name": asset_name,
        "object_uri": (f"s3://eodata/Sentinel-3/SLSTR/SL_2_FRP___/2026/07/09/{asset_name}.nc"),
        "media_type": "application/netcdf",
        "file_size_bytes": source_path.stat().st_size,
        "file_checksum": sha256(source_path.read_bytes()).hexdigest(),
    }
    artifact = _observation_artifact(
        collection_key=collection_key,
        processor="sentinel3_frp_v1",
        assets=[asset_payload],
        acquired_at=datetime(2026, 7, 9, 10, tzinfo=UTC),
        resolution_m=1_000,
    )
    durable = _observation_durable(artifact)
    publisher = _ObservationPublisher()
    reader = _LocalObservationReader(frp_path=source_path)

    receipt = SatelliteObservationCpuWorker(
        repository=_Repository(durable),
        asset_reader=reader,
        publisher=publisher,
        minimum_frp_confidence=0.3,
    ).run(durable.event.event_id, artifact.artifact_revision_id)

    assert receipt.status == "completed"
    assert reader.frp_ephemeral_path is not None
    assert not reader.frp_ephemeral_path.exists()
    payload = publisher.payloads[0]
    assert payload["raw_satellite_content_stored"] is False
    assert len(payload["observations"]) == 1
    observation = payload["observations"][0]
    assert observation["geometry_geojson"] == {"type": "Point", "coordinates": [5.37, 44.75]}
    assert observation["metrics"]["frp_mwir_mw"] == 18.5
    assert observation["metrics"]["provider_confidence"] == 0.9
    assert "object_uri" not in str(payload)
