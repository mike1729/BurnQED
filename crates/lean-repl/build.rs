fn main() {
    println!(
        "cargo:rustc-env=LEAN_REPL_MANIFEST_DIR={}",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    println!("cargo:rerun-if-changed=build.rs");
}
