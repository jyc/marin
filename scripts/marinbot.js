// If we can't read marinbot.json assume these are the maintainers.
const FALLBACK_MAINTAINERS = ["dlwh"];

// Parse command from comment body
function parseCommand(body, commandPrefix) {
  const line = body
    .split(/\r?\n/)
    .map((s) => s.trim())
    .find((s) => s.startsWith(commandPrefix));
  if (!line) {
    return null;
  }

  const rest = line.substring(commandPrefix.length).trim();
  const tokens = rest.split(/\s+/).filter(Boolean);
  const { named, positional } = parseArgs(tokens);

  return { line, tokens, named, positional };
}

// Parse --key value arguments
function parseArgs(tokens) {
  const named = {};
  const positional = [];

  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].startsWith("--")) {
      const key = tokens[i].substring(2);
      if (i + 1 < tokens.length && !tokens[i + 1].startsWith("--")) {
        named[key] = tokens[i + 1];
        i++;
      } else {
        named[key] = true;
      }
    } else {
      positional.push(tokens[i]);
    }
  }

  return { named, positional };
}

async function loadMaintainers({ github, owner, repo, defBranch }) {
  try {
    const cfg = await github.rest.repos.getContent({
      owner,
      repo,
      path: "marinbot.json",
      ref: defBranch,
    });
    const content = Buffer.from(cfg.data.content, cfg.data.encoding).toString(
      "utf8"
    );
    const config = JSON.parse(content);
    return config.maintainers || FALLBACK_MAINTAINERS;
  } catch (e) {
    console.log(
      `Could not read maintainer config marinbot.json, falling back to ${JSON.stringify(
        FALLBACK_MAINTAINERS
      )}`
    );
    return FALLBACK_MAINTAINERS;
  }
}

async function validateMaintainer({
  github,
  context,
  owner,
  repo,
  actor,
  maintainers,
}) {
  if (!maintainers.includes(actor)) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ @${actor} is not a maintainer`,
    });
    throw new Error(`@${actor} is not a maintainer.`);
  }
}

async function validatePullRequest({ github, context, owner, repo }) {
  if (!context.payload.issue || !context.payload.issue.pull_request) {
    if (context.payload.issue) {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: context.payload.issue.number,
        body: `❌ Only works on pull requests`,
      });
    }
    throw new Error("not pull request comment");
  }
}

async function handleStop({ github, context, core, githubOutput }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  await validatePullRequest({ github, context, owner, repo });
  const repoInfo = await github.rest.repos.get({ owner, repo });
  const defBranch = repoInfo.data.default_branch;
  const maintainers = await loadMaintainers({ github, owner, repo, defBranch });
  const actor = context.payload.comment.user.login;
  await validateMaintainer({
    github,
    context,
    owner,
    repo,
    actor,
    maintainers,
  });
  const body = context.payload.comment.body || "";
  const parsed = parseCommand(body, "@marinbot stop ");

  if (!parsed) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ Invalid command. Use: \`@marinbot stop --cluster <path> <job_id>\``,
    });
    throw new Error(`invalid command: ${body}`);
  }

  const { named, positional } = parsed;

  const clusterPath = named.cluster;
  const jobId = positional[0];

  if (!clusterPath) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ Missing --cluster. Use: \`@marinbot stop --cluster <path> <job_id>\``,
    });
    throw new Error(`missing cluster: ${body}`);
  }

  if (!jobId) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ Missing job ID. Use: \`@marinbot stop --cluster <path> <job_id>\``,
    });
    throw new Error(`missing job id: ${body}`);
  }
  const prNumber = context.payload.issue.number;
  const pr = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: prNumber,
  });

  const result = {
    pr_number: String(prNumber),
    head_ref: pr.data.head.ref,
    sha: pr.data.head.sha,
    cluster_path: clusterPath,
    job_id: jobId,
    actor: actor,
  };

  // Write directly to GITHUB_OUTPUT
  if (!githubOutput) {
    throw new Error("GITHUB_OUTPUT environment variable is required");
  }
  const fs = require("fs");
  const outputs = Object.entries(result)
    .map(([key, value]) => `${key}=${value}`)
    .join("\n");
  fs.appendFileSync(githubOutput, outputs + "\n");

  return result;
}

async function handleRayRun({ github, context, core, githubOutput }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const body = context.payload.comment.body || "";
  await validatePullRequest({ github, context, owner, repo });
  const repoInfo = await github.rest.repos.get({ owner, repo });
  const defBranch = repoInfo.data.default_branch;
  const maintainers = await loadMaintainers({ github, owner, repo, defBranch });
  const actor = context.payload.comment.user.login;
  await validateMaintainer({
    github,
    context,
    owner,
    repo,
    actor,
    maintainers,
  });
  const parsed = parseCommand(body, "@marinbot ray_run ");

  if (!parsed) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ Invalid command. Use: \`@marinbot ray_run --cluster <path> <module>\``,
    });
    throw new Error(`invalid command: ${body}`);
  }

  const { line, named, positional } = parsed;
  const moduleName = positional[positional.length - 1];

  if (!moduleName) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ Missing module. Use: \`@marinbot ray_run --cluster <path> <module>\``,
    });
    throw new Error(`missing module: ${body}`);
  }

  const clusterPath = named.cluster;

  if (!clusterPath) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.payload.issue.number,
      body: `❌ Missing --cluster. Use: \`@marinbot ray_run --cluster <path> <module>\``,
    });
    throw new Error(`missing cluster: ${body}`);
  }
  const prNumber = context.payload.issue.number;
  const pr = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: prNumber,
  });
  const rayArgsTokens = [];
  for (const [key, value] of Object.entries(named)) {
    rayArgsTokens.push(`--${key}`);
    if (value !== true) {
      rayArgsTokens.push(value);
    }
  }
  rayArgsTokens.push(...positional.slice(0, -1));

  const result = {
    pr_number: String(prNumber),
    head_ref: pr.data.head.ref,
    sha: pr.data.head.sha,
    module: moduleName,
    cluster_path: clusterPath,
    ray_args: rayArgsTokens.join(" "),
    full_command: line,
    actor: actor,
  };

  // Write directly to GITHUB_OUTPUT
  if (!githubOutput) {
    throw new Error("GITHUB_OUTPUT environment variable is required");
  }
  const fs = require("fs");
  const outputs = Object.entries(result)
    .map(([key, value]) => `${key}=${value}`)
    .join("\n");
  fs.appendFileSync(githubOutput, outputs + "\n");

  return result;
}

async function handle({ github, context, core }) {
  const body = context.payload.comment?.body || "";
  const githubOutput = process.env.GITHUB_OUTPUT;

  if (!githubOutput) {
    throw new Error("GITHUB_OUTPUT environment variable is required");
  }

  const fs = require("fs");

  // Determine command type
  let command = "unknown";
  if (body.includes("@marinbot stop")) {
    command = "stop";
  } else if (body.includes("@marinbot ray_run")) {
    command = "ray_run";
  }

  // Write command type to output
  fs.appendFileSync(githubOutput, `command=${command}\n`);

  // Handle the appropriate command
  try {
    if (command === "stop") {
      await handleStop({ github, context, core, githubOutput });
    } else if (command === "ray_run") {
      await handleRayRun({ github, context, core, githubOutput });
    } else {
      throw new Error("Unknown command");
    }
  } catch (error) {
    // Re-throw to let GitHub Actions handle the failure
    throw error;
  }
}

module.exports = {
  handle,
};
