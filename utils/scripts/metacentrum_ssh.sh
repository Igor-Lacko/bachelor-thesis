#!bin/bash
mkdir -p "$SCRATCHDIR/.ssh"
chmod 700 "$SCRATCHDIR/.ssh"

rm -f "$SCRATCHDIR/.ssh/server_key" "$SCRATCHDIR/.ssh/server_key.pub"
ssh-keygen -f "$SCRATCHDIR/.ssh/server_key" -N "" -C "VSCode" -q -t ed25519
echo "SSH key generated"

# to $SCRATCHDIR/.ssh/authorized_keys, add public key from step 1
touch "$SCRATCHDIR/.ssh/authorized_keys"
chmod 600 "$SCRATCHDIR/.ssh/authorized_keys"

PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAuQO3nvORQVauj9s0WRrMncf5FdF6I3g1EdHIkuQ5zc iginko@zeus'
grep -q "$PUBKEY" "$SCRATCHDIR/.ssh/authorized_keys" || echo "$PUBKEY" >> "$SCRATCHDIR/.ssh/authorized_keys"

echo "Public key added to authorized_keys"

# make sure that the chosen port is free
PORT=14000
netstat -tan |grep -q "0.0.0.0:$PORT " && { echo "port $PORT is busy" >&2; exit 1; }

# on the computational node, start running ssh server demon that listens on $PORT port
echo "Starting SSH server"
/usr/sbin/sshd -D -p $PORT \
    -o "HostKey $SCRATCHDIR/.ssh/server_key" \
    -o "PermitRootLogin no" \
    -o "PasswordAuthentication no" \
    -o "ChallengeResponseAuthentication no" \
    -o "AllowTcpForwarding yes" \
    -o "AuthorizedKeysFile $SCRATCHDIR/.ssh/authorized_keys" &

echo "SSH running on $(hostname):$PORT"
echo "Connect to:" hostname