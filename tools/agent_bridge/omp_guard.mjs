// OhMyPi (omp) tool-call guard for unattended agent-bridge runs.
//
// omp's file tools honour absolute paths, "~" and "..", and `write` can address `xd://` tool
// devices, so on their own they are not confined to the task worktree.  The bridge loads this
// module with `--trusted-extension` and gives it a JSON policy (LAWNBERRY_BRIDGE_GUARD_POLICY)
// that lives outside the worktree.  Every tool call is checked before it executes:
//
//   * only tools named in `policy.tools` (plus the hidden `yield`/`think`) may run;
//   * every path argument must be a plain POSIX path (see PLAIN_PATH: omp rewrites a path before
//     it opens it - leading "@" or ":", quotes, whitespace/comma/semicolon lists, "~", braces,
//     file:// - and the guard must never approve a different file than omp opens), must stay
//     inside `policy.root` (symlinks resolved), must not be a URL or `xd://` device, must not
//     contain `..` segments and must not match `read_deny`; the path as written and its
//     symlink-resolved form are both matched against `read_deny`, `write_deny` and the allow sets;
//   * write-class tools must also match every set in `write_allow_sets` and no `write_deny` glob,
//     must name their files in path arguments (edit syntaxes that carry paths inside a text
//     `input` - hashline, apply_patch, sloppy - are refused) and are capped in total payload bytes;
//   * `bash`, when enabled at all, may only run command lines listed verbatim in `bash_exact`,
//     and its path arguments (`cwd`) are checked like any other.
//
// A missing or invalid policy blocks everything (fail closed).  Decisions are appended to
// `policy.audit_log` as JSON lines.  The glob dialect is identical to tools/agent_bridge/globs.py.

import * as fs from "node:fs";
import * as path from "node:path";

export const PREFIX = "[agent-bridge guard]";
const ALWAYS_ALLOWED = new Set(["yield", "think"]);
const WRITE_TOOLS = new Set(["edit", "write"]);
const PATH_KEYS = new Set([
	"path", "paths", "file", "files", "file_path", "filePath", "filepath",
	"to", "moveTo", "from", "source", "dest", "destination", "rename", "cwd", "dir", "directory", "root",
]);
const PATTERN_PATH_TOOLS = new Set(["glob", "find"]);
const MAX_AUDIT_LINES = 5000;

// What a path argument may consist of.  omp rewrites a path before it reaches the disk: it strips
// a leading "@" or ":", removes surrounding quotes, splits one string into several paths on
// whitespace, "," and ";", expands "~" and braces, maps Unicode spaces to " " and turns file://
// URLs into paths (omp tools/path-utils.ts: expandPath, splitDelimitedPathEntry).  The guard
// reads the text literally, so each of those made it approve a different file than omp then
// opened ("@/etc/passwd" is the in-worktree file "@/etc/passwd" to the guard).  Instead of
// modelling omp's pipeline, everything that is not a plain POSIX path is refused.
const PLAIN_PATH = /^[A-Za-z0-9._+\-\/]+$/;
const GLOB_PATH = /^[A-Za-z0-9._+\-\/*?[\]]+$/;
const SELECTOR = /^[A-Za-z0-9,+:\-]+$/;

export function globToRegExp(pattern) {
	let out = "";
	let i = 0;
	const n = pattern.length;
	const escape = (ch) => ch.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&");
	while (i < n) {
		const ch = pattern[i];
		if (ch === "*") {
			if (pattern.startsWith("**", i)) {
				const j = i + 2;
				if (j < n && pattern[j] === "/") {
					out += "(?:.*/)?";
					i = j + 1;
					continue;
				}
				if (out.endsWith("/")) {
					out = out.slice(0, -1) + "(?:/.*)?";
				} else {
					out += ".*";
				}
				i = j;
				continue;
			}
			out += "[^/]*";
		} else if (ch === "?") {
			out += "[^/]";
		} else {
			out += escape(ch);
		}
		i += 1;
	}
	return new RegExp("^" + out + "$", "s");
}

export function matchAny(rel, patterns) {
	return patterns.some((p) => globToRegExp(p).test(rel));
}

function isInside(root, candidate) {
	return candidate === root || candidate.startsWith(root + path.sep);
}

// realpath of the deepest existing ancestor, with the not-yet-existing tail re-appended
function realResolve(abs) {
	let current = abs;
	const tail = [];
	for (;;) {
		try {
			const real = fs.realpathSync(current);
			return tail.length ? path.join(real, ...tail.reverse()) : real;
		} catch {
			const parent = path.dirname(current);
			if (parent === current) return abs;
			tail.push(path.basename(current));
			current = parent;
		}
	}
}

export function collectPaths(input, toolName) {
	const found = [];
	const walk = (value, key) => {
		if (value === null || value === undefined) return;
		if (Array.isArray(value)) {
			for (const item of value) walk(item, key);
			return;
		}
		if (typeof value === "object") {
			for (const [k, v] of Object.entries(value)) walk(v, k);
			return;
		}
		if (typeof value === "string") {
			if (key && (PATH_KEYS.has(key) || (key === "pattern" && PATTERN_PATH_TOOLS.has(toolName)))) {
				found.push(value);
			}
		}
	};
	walk(input, null);
	return found;
}

// Why a raw path argument cannot be verified, or null when it is a plain path (see PLAIN_PATH).
function unverifiablePath(part, allowGlob) {
	const colon = part.indexOf(":");
	const body = colon > 0 ? part.slice(0, colon) : part;
	if (!(allowGlob ? GLOB_PATH : PLAIN_PATH).test(body)) {
		return allowGlob
			? "path uses characters the guard cannot verify (only letters, digits and . _ + - / * ? [ ])"
			: "path uses characters the guard cannot verify (only letters, digits and . _ + - /)";
	}
	if (colon > 0 && !SELECTOR.test(part.slice(colon + 1))) {
		return "path selector uses characters the guard cannot verify";
	}
	if (body.split("/").includes("..")) {
		return "parent-directory segments (..) are not allowed; name a path below the worktree root";
	}
	return null;
}

// Returns {ok, rels?, reals?, reason?} for one raw path argument.
export function resolveWithin(raw, root, realRoot, cwd, allowGlob = false) {
	if (typeof raw !== "string" || raw.length === 0) return { ok: false, reason: "empty path argument" };
	if (raw.includes("\0")) return { ok: false, reason: "NUL in path" };
	if (/^[A-Za-z][A-Za-z0-9+.\-]*:\/\//.test(raw)) {
		return { ok: false, reason: `URL or device path is not allowed: ${raw.slice(0, 60)}` };
	}
	if (raw.startsWith("~")) return { ok: false, reason: "home-relative paths are not allowed" };
	const rels = [];
	const reals = [];
	for (const part of raw.split(";")) {
		if (part === "") continue;
		const variants = new Set([part]);
		const colon = part.indexOf(":");
		if (colon > 0) variants.add(part.slice(0, colon)); // file.ext:10-20 selectors
		for (const variant of variants) {
			const abs = path.resolve(cwd, variant);
			if (!isInside(root, abs)) {
				return { ok: false, reason: `path escapes the task worktree: ${part.slice(0, 80)}` };
			}
			const real = realResolve(abs);
			if (!isInside(realRoot, real)) {
				return { ok: false, reason: `path resolves outside the task worktree: ${part.slice(0, 80)}` };
			}
			rels.push(path.relative(root, abs).split(path.sep).join("/") || ".");
			reals.push(path.relative(realRoot, real).split(path.sep).join("/") || ".");
		}
		const why = unverifiablePath(part, allowGlob);
		if (why) return { ok: false, reason: `${why}: ${part.slice(0, 80)}` };
	}
	if (rels.length === 0) return { ok: false, reason: "empty path argument" };
	return { ok: true, rels, reals };
}

export function validatePolicy(policy) {
	return (
		policy &&
		typeof policy === "object" &&
		typeof policy.root === "string" &&
		path.isAbsolute(policy.root) &&
		Array.isArray(policy.tools) &&
		Array.isArray(policy.write_allow_sets) &&
		Array.isArray(policy.write_deny) &&
		Array.isArray(policy.read_deny)
	);
}

// True when any object in the arguments has a string-valued `input` key: omp's hashline,
// apply_patch and sloppy edit syntaxes put file headers and MV/REM/rename targets in that text.
function hasInlineInput(value) {
	if (Array.isArray(value)) return value.some(hasInlineInput);
	if (value && typeof value === "object") {
		return Object.entries(value).some(([key, item]) => (key === "input" && typeof item === "string") || hasInlineInput(item));
	}
	return false;
}

// Total UTF-8 size of every string in the arguments except the path-valued ones.
function payloadBytes(value, key) {
	if (typeof value === "string") return key && PATH_KEYS.has(key) ? 0 : Buffer.byteLength(value, "utf8");
	if (Array.isArray(value)) return value.reduce((total, item) => total + payloadBytes(item, key), 0);
	if (value && typeof value === "object") {
		return Object.entries(value).reduce((total, [name, item]) => total + payloadBytes(item, name), 0);
	}
	return 0;
}

// Pure decision function: {allow: boolean, reason?: string, paths: string[]}
export function decide(policy, event, cwdFallback) {
	const tool = String(event?.toolName ?? "");
	const input = event?.input && typeof event.input === "object" ? event.input : {};
	if (!validatePolicy(policy)) return { allow: false, reason: "guard policy missing or invalid", paths: [] };
	if (ALWAYS_ALLOWED.has(tool)) return { allow: true, paths: [] };
	if (!policy.tools.includes(tool)) return { allow: false, reason: `tool '${tool}' is not enabled`, paths: [] };

	const root = path.resolve(policy.root);
	let realRoot;
	try {
		realRoot = fs.realpathSync(root);
	} catch {
		return { allow: false, reason: "task worktree is not accessible", paths: [] };
	}
	const cwd = typeof cwdFallback === "string" && cwdFallback ? cwdFallback : root;

	if (tool === "bash") {
		const command = typeof input.command === "string" ? input.command.trim() : "";
		const exact = Array.isArray(policy.bash_exact) ? policy.bash_exact : [];
		if (!exact.includes(command)) return { allow: false, reason: "command is not on the allowlist", paths: [] };
		// an allowlisted command line still has to keep its path arguments (cwd, ...) inside the worktree
	}

	const raws = collectPaths(input, tool);
	const isWrite = WRITE_TOOLS.has(tool);
	if (isWrite && hasInlineInput(input)) {
		return {
			allow: false,
			reason: "edit syntax that names files inside its text cannot be verified; use path with old_string/new_string, or write",
			paths: [],
		};
	}
	if (isWrite && raws.length === 0) {
		return { allow: false, reason: "write call without a recognisable path (failing closed)", paths: [] };
	}
	const rels = [];
	const reals = [];
	for (const raw of raws) {
		const resolved = resolveWithin(raw, root, realRoot, cwd, !isWrite && tool !== "bash");
		if (!resolved.ok) return { allow: false, reason: resolved.reason, paths: rels };
		rels.push(...resolved.rels);
		reals.push(...resolved.reals);
	}
	// The lexical path and the symlink-resolved path must both pass: a directory symlink inside an
	// allowed folder must not lead to a protected, out-of-scope or read-denied file.
	for (const rel of new Set([...rels, ...reals])) {
		if (rel !== "." && matchAny(rel, policy.read_deny)) {
			return { allow: false, reason: `path is not readable by agents: ${rel}`, paths: rels };
		}
		if (!isWrite) continue;
		if (rel === ".") return { allow: false, reason: "cannot write the repository root", paths: rels };
		if (matchAny(rel, policy.write_deny)) {
			return { allow: false, reason: `path is protected and needs a human change: ${rel}`, paths: rels };
		}
		for (const set of policy.write_allow_sets) {
			if (!matchAny(rel, set)) {
				return { allow: false, reason: `path is outside the task's allowed scope: ${rel}`, paths: rels };
			}
		}
		try {
			if (fs.lstatSync(path.join(root, rel)).isSymbolicLink()) {
				return { allow: false, reason: `refusing to write through a symbolic link: ${rel}`, paths: rels };
			}
		} catch {
			// target does not exist yet: fine
		}
	}
	const maxBytes = Number(policy.max_write_bytes ?? 0);
	if (isWrite && maxBytes > 0 && payloadBytes(input, null) > maxBytes) {
		return { allow: false, reason: `write exceeds the ${maxBytes} byte limit`, paths: rels };
	}
	return { allow: true, paths: rels };
}

function loadPolicy() {
	const file = process.env.LAWNBERRY_BRIDGE_GUARD_POLICY;
	if (!file) return null;
	try {
		const policy = JSON.parse(fs.readFileSync(file, "utf8"));
		return validatePolicy(policy) ? policy : null;
	} catch {
		return null;
	}
}

export default function guard(pi) {
	const policy = loadPolicy();
	let audited = 0;
	const audit = (tool, decision) => {
		if (!policy?.audit_log || audited >= MAX_AUDIT_LINES) return;
		audited += 1;
		try {
			fs.appendFileSync(
				policy.audit_log,
				JSON.stringify({
					ts: new Date().toISOString(),
					tool,
					allow: decision.allow,
					reason: decision.reason ?? null,
					paths: decision.paths ?? [],
				}) + "\n",
			);
		} catch {
			// auditing must never break the run
		}
	};
	pi.on("tool_call", async (event, ctx) => {
		let decision;
		try {
			decision = decide(policy, event, ctx?.cwd);
		} catch (error) {
			decision = { allow: false, reason: `guard error: ${String(error).slice(0, 120)}`, paths: [] };
		}
		audit(String(event?.toolName ?? ""), decision);
		if (!decision.allow) return { block: true, reason: `${PREFIX} ${decision.reason}` };
		return undefined;
	});
}
