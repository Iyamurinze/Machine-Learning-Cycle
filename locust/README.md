# Flood request simulation

Procedure for measuring how the deployed model behaves under a flood of
requests, and how that changes with the number of API containers.

## What is being measured

The `/predict` endpoint, weighted to 10 of every 16 requests. Each request
posts a real cell image from `data/test/` and **asserts the returned class is
correct** — a container serving HTTP 200 with a broken or partially loaded
model is recorded as a failure rather than a success, which a liveness-only
test would miss entirely.

## Setup

Locust needs the test images present locally, since it posts real cells:

```bash
python src/data_acquisition.py     # if not already done
pip install locust
```

Bring up the stack. The API sits behind nginx on port 8080, so the number of
replicas can change while Locust keeps hitting one address:

```bash
docker compose up -d --build
```

## Running the three measurements

Each run: 100 users, spawning at 10/second, for 2 minutes.

```bash
mkdir -p locust/reports

# --- 1 container ---
docker compose up -d --scale api=1
sleep 45                                  # let TensorFlow finish importing
locust -f locust/locustfile.py --host http://localhost:8080 \
    --users 100 --spawn-rate 10 --run-time 2m --headless \
    --html locust/reports/report_1container.html \
    --csv locust/reports/run_1container

# --- 2 containers ---
docker compose up -d --scale api=2
sleep 45
locust -f locust/locustfile.py --host http://localhost:8080 \
    --users 100 --spawn-rate 10 --run-time 2m --headless \
    --html locust/reports/report_2containers.html \
    --csv locust/reports/run_2containers

# --- 4 containers ---
docker compose up -d --scale api=4
sleep 45
locust -f locust/locustfile.py --host http://localhost:8080 \
    --users 100 --spawn-rate 10 --run-time 2m --headless \
    --html locust/reports/report_4containers.html \
    --csv locust/reports/run_4containers
```

The `sleep 45` matters. A freshly started container has not yet imported
TensorFlow or loaded the model, so its first request takes seconds. Starting
the load immediately would fold that one-off cost into the steady-state latency
figures and make a larger replica count look artificially *worse*, since more
replicas means more cold starts.

## Interactive mode

To watch the request rate and latency live, or to drive load by hand:

```bash
locust -f locust/locustfile.py --host http://localhost:8080
```

Then open <http://localhost:8089>.

## Reading the results

Each run writes an HTML report and a set of CSVs to `locust/reports/`. The
figures that go into the project README come from `run_*_stats.csv`:

| Column | Meaning |
|---|---|
| `Request Count` | total requests completed |
| `Failure Count` | failures, including wrong-class responses |
| `Median Response Time` | the typical user's experience |
| `95%` | tail latency — what the unluckiest 1 in 20 users sees |
| `Requests/s` | throughput |

Median latency describes the typical case; **p95 is the number that matters for
capacity planning**, because it is where queueing shows up first. Under a
saturated single container, the median can stay respectable while p95 climbs
sharply — the queue is forming but has not yet reached the middle of the
distribution.

## A note on interpreting the scaling

Every replica is capped at 1 CPU (`docker-compose.yml`) with TensorFlow pinned
to a single thread. Without that cap, each container would try to use all host
cores, replicas would contend for the same CPUs, and adding containers would
show little or no improvement — measuring thread thrash rather than
concurrency.

Scaling is also bounded by the host: on a machine with N cores, throughput
stops improving once the replica count approaches N, regardless of how many
containers are started. Results should be read against the core count of the
machine that produced them, which is recorded alongside the results table in
the project README.
