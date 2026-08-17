#!/usr/bin/env bash
set -u
hosts=(10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64 10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72 10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78)
tmp=$(mktemp -d)
for host in "${hosts[@]}"; do
  (
    ssh -o BatchMode=yes -o ConnectTimeout=7 "$host" "bash --noprofile --norc -c '
      cpu=\$(awk -v n=\"\$(nproc)\" \"{printf \\\"%.1f\\\",100*\\\$1/n}\" /proc/loadavg)
      io=\$(awk \"/^some /{for(i=1;i<=NF;i++)if(\\\$i~/^avg10=/){split(\\\$i,a,\\\"=\\\");print a[2]}}\" /proc/pressure/io)
      collectors=\$(pgrep -fc \"[c]ollect_universal_ppo_rollouts.py\" || true)
      workers=\$(pgrep -fc \"[r]un_async_ppo_rollout_worker.py\" || true)
      a08=\$(pgrep -fc \"[c]ollect_universal_ppo_rollouts.py.*a08_dipplin_seaking\" || true)
      cores=\$(nproc)
      mem=\$(free -g | awk \"/^Mem:/{print \\\$7}\")
      echo \"$host cpu=\$cpu io=\$io cores=\$cores availMemGiB=\$mem collectors=\$collectors workers=\$workers a08=\$a08\"
    '" > "$tmp/${host##*.}" 2>/dev/null || true
  ) &
done
wait
find "$tmp" -type f -size +0c -print0 | sort -z | xargs -0 -r cat
rm -rf "$tmp"
