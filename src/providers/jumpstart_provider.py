import json, time, logging
from typing import Iterable, Tuple, Optional
from botocore.exceptions import ClientError
from .base import Provider

try:
    from sagemaker import Session
    from sagemaker.jumpstart.model import JumpStartModel
except Exception:
    Session = None
    JumpStartModel = None

log = logging.getLogger("jumpstart")


class JumpStartProvider(Provider):
    name = "jumpstart"

    def __init__(
        self,
        region: str = "us-east-2",
        exec_role_arn: Optional[str] = None,
        default_instance: str = "ml.g5.2xlarge",
    ):
        self.region = region
        self.exec_role_arn = exec_role_arn
        self.default_instance = default_instance
        self._sm_sess = None
        try:
            import boto3

            js = boto3.session.Session(region_name=self.region)
            self._sm = js.client("sagemaker")
            self._rt = js.client("sagemaker-runtime")
            self._sm_sess = Session(boto_session=js) if Session else None
        except Exception:
            self._sm = None
            self._rt = None

    def is_ready(self) -> bool:
        return (
            JumpStartModel is not None
            and self._sm is not None
            and self._sm_sess is not None
        )

    def list_models(self) -> Iterable[str]:
        # JumpStart discovery can be expensive; keep as empty and use explicit model IDs when needed.
        return []

    def _pick_fallback_instance(self, model_id: str) -> str:
        mid = model_id.lower()
        if "120b" in mid:
            return "ml.p5.48xlarge"
        if "20b" in mid or "8b" in mid:
            return "ml.g5.2xlarge"
        return self.default_instance

    def run(self, prompt: str, model_id: str) -> Tuple[str, float]:
        if not self.is_ready():
            raise RuntimeError("SageMaker JumpStart not initialized")
        t0 = time.time()
        model = JumpStartModel(
            model_id=model_id, role=self.exec_role_arn, sagemaker_session=self._sm_sess
        )
        endpoint_name = f"jumpstart-ephemeral-{int(time.time())}"
        instance_type = self._pick_fallback_instance(model_id)
        accept_eula = "llama" in model_id.lower()

        try:
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                accept_eula=accept_eula,
            )
        except ClientError as e:
            if "failed to satisfy constraint" in str(e):
                fb = self._pick_fallback_instance(model_id)
                log.warning(f"Retrying deploy with fallback instance {fb}")
                predictor = model.deploy(
                    initial_instance_count=1,
                    instance_type=fb,
                    endpoint_name=endpoint_name,
                    accept_eula=accept_eula,
                )
            else:
                raise

        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128}}
        try:
            resp = self._rt.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            body = resp["Body"].read().decode("utf-8", errors="ignore")
        finally:
            try:
                self._sm.delete_endpoint(EndpointName=endpoint_name)
                self._sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
            except Exception:
                log.warning("Cleanup failed or endpoint already gone.")
        return body, time.time() - t0
