# Minimal clean-room image for the flow->anneal TRUE-ISOLATION canary.
# Just node + the claude CLI, pinned to the host version. NO flow config / hooks /
# continuity / state — isolation by construction. The candidate + me + partnership ride
# in as PROMPT context only; /root/.claude holds ONLY the OAuth login (from a one-time
# `claude` login into a dedicated volume).
FROM node:22-slim
RUN npm install -g @anthropic-ai/claude-code@2.1.158 \
 && claude --version
ENTRYPOINT ["/bin/bash"]
