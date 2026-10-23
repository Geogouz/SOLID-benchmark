//! Embeds in every executable of this package the identity of the sources it is built from, a SHA-256 digest of the
//! files of its own source trees rather than the commit, so that a commit that changes records or documents alone
//! leaves every executable as it is. The files are those SOURCES names below the repository root. The digest is the
//! SHA-256 of the lines "<SHA-256 of the file>  <path>", one per file in the order of the paths and each ending in a
//! newline, the listing of sha256sum. research/benchmarks/datasets/isolation.py computes the same digest, so the
//! scripts that build an executable can check the identity it prints with --identity.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// The files the executables are built from, relative to the repository root: the toolchain, the cargo configuration
/// with the build settings of OpenBLAS, the manifests and lock files of both workspaces, and the sources of the engine,
/// of the research crates and of the executables that live beside their drivers. An entry with extensions is a
/// directory standing for every file below it with one of them, symbolic links left out, and an entry without is one
/// file.
const SOURCES: &[(&str, &[&str])] = &[
    ("rust-toolchain.toml", &[]),
    (".cargo/config.toml", &[]),
    ("runtime/Cargo.toml", &[]),
    ("runtime/engine/Cargo.toml", &[]),
    ("runtime/engine/src", &["rs"]),
    ("research/Cargo.toml", &[]),
    ("research/Cargo.lock", &[]),
    ("research/cli/Cargo.toml", &[]),
    ("research/cli/build.rs", &[]),
    ("research/cli/src", &["rs"]),
    ("research/evaluation/Cargo.toml", &[]),
    ("research/evaluation/src", &["rs"]),
    ("research/io/Cargo.toml", &[]),
    ("research/io/src", &["rs"]),
    ("research/methods/Cargo.toml", &[]),
    ("research/methods/src", &["rs"]),
    ("research/benchmarks/datasets", &["rs"]),
    ("research/solid", &["rs"]),
];

/// SHA-256 of FIPS 180-4, written out so that the build script needs no dependency.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&(data.len() as u64 * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = state;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let choice = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(choice).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let majority = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(majority);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (word, value) in state.iter_mut().zip(v) {
            *word = word.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The path of a file below the root with forward slashes.
fn relative(root: &Path, path: &Path) -> String {
    let below = path.strip_prefix(root).expect("a source lies below the repository root");
    below.components().map(|part| part.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")
}

/// Every file below a directory with one of the extensions, symbolic links left out.
fn walk(root: &Path, directory: &Path, extensions: &[&str], found: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let (Ok(kind), path) = (entry.file_type(), entry.path()) else {
            continue;
        };
        if kind.is_dir() {
            walk(root, &path, extensions, found);
        } else if kind.is_file()
            && path.extension().and_then(|extension| extension.to_str()).is_some_and(|extension| extensions.contains(&extension))
        {
            found.push((relative(root, &path), path));
        }
    }
}

/// Every source file is watched on its own, so an edit of another file of the same directories, a Python driver among
/// them, leaves the script as it is. A new source file enters a build only through an edit of a watched one, the
/// module that declares it or the manifest that names it, which runs the script again.
fn main() {
    let manifest = env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set");
    let root = Path::new(&manifest).join("../..");
    let mut found = Vec::new();
    for (name, extensions) in SOURCES {
        let path = root.join(name);
        if extensions.is_empty() {
            println!("cargo:rerun-if-changed={}", path.display());
            if path.is_file() {
                found.push((name.to_string(), path));
            }
        } else {
            walk(&root, &path, extensions, &mut found);
        }
    }
    found.sort();
    for (_, path) in &found {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let listing: String = found
        .iter()
        .map(|(name, path)| format!("{}  {name}\n", hex(&sha256(&fs::read(path).expect("a readable source")))))
        .collect();
    println!("cargo:rustc-env=SOLID_SOURCE_IDENTITY={}", hex(&sha256(listing.as_bytes())));
}
