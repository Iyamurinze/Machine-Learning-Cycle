"""Locust load test for the malaria classification API.

Simulates users hitting the deployed service, weighted toward the endpoint
that actually costs something. Run against the nginx balancer so the request
load is spread across however many API containers are running:

    locust -f locust/locustfile.py --host http://localhost:8080

Headless, as used for the recorded results:

    locust -f locust/locustfile.py --host http://localhost:8080 \
        --users 100 --spawn-rate 10 --run-time 2m --headless \
        --html locust/reports/report_1container.html

See locust/README.md for the full procedure.
"""

from __future__ import annotations

import random
from pathlib import Path

from locust import HttpUser, between, events, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "test"
CLASS_NAMES = ("Parasitized", "Uninfected")

# Images are read into memory once at startup rather than per request. Reading
# from disk inside a task would measure the load generator's filesystem
# instead of the API, and would cap throughput on the client side.
IMAGE_POOL: list[tuple[str, bytes, str]] = []


@events.test_start.add_listener
def load_images(environment, **kwargs) -> None:
    """Preload a pool of real test images before the run starts."""
    global IMAGE_POOL

    if not TEST_DIR.is_dir():
        print(
            f"WARNING: {TEST_DIR} not found. Run "
            f"`python src/data_acquisition.py` first — the load test needs "
            f"real images to post."
        )
        return

    pool = []
    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.png"))[:100]:
            pool.append((path.name, path.read_bytes(), class_name))

    IMAGE_POOL = pool
    print(f"Loaded {len(IMAGE_POOL)} images into the request pool.")


class MalariaAPIUser(HttpUser):
    """A user of the deployed classifier.

    `between(1, 3)` models humans uploading cells rather than a pure
    throughput hammer — the goal is to characterise latency under realistic
    concurrency, and a zero-wait loop would only measure how fast the client
    can saturate one socket.
    """

    wait_time = between(1, 3)

    @task(10)
    def predict(self) -> None:
        """The expensive path: a full forward pass through the CNN."""
        if not IMAGE_POOL:
            return

        filename, payload, true_label = random.choice(IMAGE_POOL)

        with self.client.post(
            "/predict",
            files={"file": (filename, payload, "image/png")},
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("Response was not JSON")
                return

            if "prediction" not in body:
                response.failure("No prediction in response")
                return

            # Correctness is asserted, not just liveness. A container serving
            # 200s with a broken or half-loaded model would otherwise show up
            # as a perfectly healthy load test.
            if body["prediction"] != true_label:
                response.failure(
                    f"Wrong class: said {body['prediction']}, expected {true_label}"
                )
                return

            response.success()

    @task(3)
    def status(self) -> None:
        """Dashboard polling — cheap, but every replica serves it."""
        self.client.get("/status", name="/status")

    @task(2)
    def health(self) -> None:
        """Load balancer health probe."""
        self.client.get("/health", name="/health")

    @task(1)
    def metrics(self) -> None:
        """Metrics read, as the dashboard's overview page does."""
        self.client.get("/metrics", name="/metrics")


class PredictOnlyUser(HttpUser):
    """Pure inference load, for isolating model throughput.

    `abstract = True` keeps this class out of the default run. Locust
    auto-discovers every concrete HttpUser subclass, so without it a plain
    `locust -f locustfile.py` would split its users across both classes and
    report two separate /predict entries — a mixed profile that is not what
    the scaling comparison intends to measure.

    To use it deliberately, drop the abstract flag or select it explicitly:
    `locust -f locust/locustfile.py PredictOnlyUser`.
    """

    abstract = True

    wait_time = between(0.5, 1.5)

    @task
    def predict(self) -> None:
        if not IMAGE_POOL:
            return
        filename, payload, _ = random.choice(IMAGE_POOL)
        self.client.post(
            "/predict",
            files={"file": (filename, payload, "image/png")},
            name="/predict [pure]",
        )
