import ast, sys

files = [
    'src/models/adaslot/train.py',
    'src/models/adaslot/model.py',
    'src/models/adaslot/decoder.py',
    'src/models/adaslot/perceptual_grouping.py',
    'src/data/__init__.py',
    'src/data/continual_tinyimagenet.py',
]
print("=== Syntax check ===")
for f in files:
    try:
        ast.parse(open(f).read())
        print(f"  [OK] {f}")
    except SyntaxError as e:
        print(f"  [ERR] {f}: {e}")

src = open('src/models/adaslot/train.py').read()
dec = open('src/models/adaslot/decoder.py').read()
ini = open('src/data/__init__.py').read()
mdl = open('src/models/adaslot/model.py').read()
pg  = open('src/models/adaslot/perceptual_grouping.py').read()

print("\n=== Fix verification ===")
checks = [
    ("Fix 1  sum/B recon loss",       "reduction='sum') / B" in src),
    ("Fix 2  mask dropped slots",     "slots_for_decode" in mdl),
    ("Fix 3  gumbel anneal",          "gumbel_temperature_schedule" in pg and "gumbel_temperature_schedule" in mdl),
    ("Fix 4  save scheduler state",   "'scheduler': scheduler.state_dict()" in src),
    ("Fix 5  optimizer persist",      "_p1_optimizer" in src),
    ("Fix 7  decoder n_upsample",     "n_upsample" in dec),
    ("Phase3 total_classes correct",  "n_tasks * args.num_classes" in src),
    ("Phase3 freeze_old fix",         "seen_classes" in src),
    ("Phase3 .cpu().numpy()",         "feats_t.cpu().numpy()" in src),
    ("AMP bfloat16 support",          "use_amp" in src),
    ("Scheduler auto-scaling",        "0.02 * num_steps" in src),
    ("No top-level avalanche import", "from avalanche" not in ini),
]

all_ok = True
for label, result in checks:
    status = "[OK]     " if result else "[MISSING]"
    if not result:
        all_ok = False
    print(f"  {status} {label}")

print()
print("ALL GOOD ✓" if all_ok else "SOME CHECKS FAILED ✗")
sys.exit(0 if all_ok else 1)
