
import os
import tokenize
import io
import itertools
import subprocess
from tempfile import NamedTemporaryFile
import csv

def generate_safe_variants(original_code):

    tokens = []
    try:
        for tok in tokenize.tokenize(io.BytesIO(original_code.encode('utf-8')).readline):
            if tok.type != tokenize.ENCODING:
                tokens.append((tokenize.tok_name[tok.type], tok.string))
    except tokenize.TokenError:
        pass

    safe_variants = []
    security_critical = {'verify', 'False'}

    allowed_indices = [
        i for i, t in enumerate(tokens)
        if t[1].strip() not in security_critical
    ]

    max_deletions = min(5, len(allowed_indices))

    for delete_num in range(1, max_deletions + 1):

        for indices in itertools.combinations(allowed_indices, delete_num):

            new_tokens = [
                t[1] for j, t in enumerate(tokens)
                if j not in set(indices)
            ]
            variant = ''.join(new_tokens)


            if not variant.strip():
                continue

            with NamedTemporaryFile('w', suffix='.py', delete=False) as f:
                f.write(variant)
                tmpfile = f.name

            # 执行安全扫描
            result = subprocess.run(['bandit', tmpfile],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
            os.unlink(tmpfile)

            if 'High: 0' in result.stdout and 'Files skipped (0)' in result.stdout and 'Medium: 0' in result.stdout and'Low: 0' in result.stdout:
                safe_variants.append(variant)

    return list(set(safe_variants))  



original_code = "r = requests.get(some_url, stream=True, verify=False)"
variants = generate_safe_variants(original_code)

# print(f"Generated {len(variants)} safe variants:")
# for i, var in enumerate(variants, 1):
#     print(f"Variant {i}:\n{var}\n")
