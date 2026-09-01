# A memory-capped environment for the demo_memory.py experiments.
#
#   container build -t pr-demo .
#   container run --rm -m 512m -v "$PWD:/work" -w /work pr-demo \
#       python polars_workflows/demo_memory.py --edges 4000000 --iterations 10
#
# The cap is the whole point.  On a large-RAM host polars never spills --  its
# out-of-core budget defaults to unlimited (see MEMORY_DIAGNOSIS.md) -- so the
# low-memory engines have nothing to prove.  Under `-m` the kernel OOM-kills a
# run that will not fit, which demo_memory.py records as OOM in its table.
FROM docker.io/library/python:3.13-slim

# Install the locked dependency set, so container numbers are comparable with
# the host's.  The venv lives outside /work, which the repo is mounted over.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock /build/
RUN pip install --no-cache-dir uv \
 && cd /build && uv sync --frozen --no-install-project \
 && rm -rf /build /root/.cache
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /work
