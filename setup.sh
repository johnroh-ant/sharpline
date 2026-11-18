#!/bin/sh
# One-time setup: creates the GitHub repo, stores two secrets, turns on the page, places the first card.
set -e
cd "$(dirname "$0")"
NAME=sharpline
command -v gh >/dev/null || { echo "Install the GitHub CLI first: brew install gh && gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login
gh auth status 2>&1 | grep -q workflow || gh auth refresh -s workflow
OWNER=$(gh api user --jq .login)
printf "Create the public repo %s/%s? Code is public; bets are encrypted. [y/N] " "$OWNER" "$NAME"; read -r yes
[ "$yes" = y ] || exit 1

OLD="$HOME/John-Roh-Personal/sports-betting/.env"
KEY=$(sed -n 's/^ODDS_API_KEY=//p' "$OLD" 2>/dev/null | tr -d "\"'")
if [ -z "$KEY" ]; then printf "Odds API key: "; stty -echo; read -r KEY; stty echo; echo; fi
PASS=$(openssl rand -hex 10)

[ -d .git ] || git init -q -b main
git config user.name "$OWNER"
git config user.email "$OWNER@users.noreply.github.com"   # keeps your own address out of a public repo
if ! git rev-parse -q --verify HEAD >/dev/null; then git add -A; git -c commit.gpgsign=false commit -qm "Sharpline"; fi
gh repo view "$OWNER/$NAME" >/dev/null 2>&1 || gh repo create "$OWNER/$NAME" --public
if gh secret list -R "$OWNER/$NAME" | grep -q LEDGER_PASSPHRASE; then
  echo "A passphrase is already set. Keeping it, because changing it would lock the existing ledger."; PASS=
else
  echo; echo "Passphrase (save it in your password manager now, it cannot be recovered): $PASS"; echo
  printf %s "$PASS" | gh secret set LEDGER_PASSPHRASE -R "$OWNER/$NAME"
fi
printf %s "$KEY" | gh secret set ODDS_API_KEY -R "$OWNER/$NAME"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$OWNER/$NAME.git"
g() { git -c credential.helper= -c credential.helper='!gh auth git-credential' "$@"; }
if g ls-remote --exit-code origin main >/dev/null; then g pull -q --rebase origin main; fi
g push -q -u origin main
gh api -X POST "repos/$OWNER/$NAME/pages" -f build_type=workflow >/dev/null 2>&1 || true
for i in 1 2 3 4 5 6; do sleep 8; gh workflow run daily.yml -R "$OWNER/$NAME" 2>/dev/null && break; done

if launchctl list 2>/dev/null | grep -q com.sportsbook.daily; then
  printf "The old model still runs on this Mac and spends the same API credits. Turn it off? [y/N] "; read -r off
  if [ "$off" = y ]; then
    launchctl bootout "gui/$(id -u)/com.sportsbook.daily" || true
    mv "$HOME/Library/LaunchAgents/com.sportsbook.daily.plist" "$HOME/Library/LaunchAgents/com.sportsbook.daily.plist.off" 2>/dev/null || true
    echo "Off. Its history is untouched. To stop the 9:32 wake-up too: sudo pmset repeat cancel"
  fi
fi
echo; echo "Dashboard (live in about 3 minutes): https://$OWNER.github.io/$NAME/"
if [ -n "$PASS" ]; then echo "Passphrase: $PASS"; fi
