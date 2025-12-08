import hashlib
import random
import string
import json
import sys
import os
import socket
import threading
import time
from queue import Queue

class Block:
    def __init__(self, sender, receiver, amount, prev_hash, depth, nonce=None, skip_mining=False):
        self.transaction = {'sender': sender, 'receiver': receiver, 'amount': amount}
        self.prev_hash = prev_hash
        self.depth = depth
    
        if nonce is not None:
            self.nonce = nonce
        elif skip_mining:
            self.nonce = "DUMMY_NONCE"
        else:
            self.nonce = self.find_nonce()
            
        self.hash = self.calculate_hash()
        self.is_decided = False

    def transaction_string(self):
        return f"{self.transaction['sender']},{self.transaction['receiver']},{self.transaction['amount']}"

    def find_nonce(self):
        print(f"[Block {self.depth}] Mining: Finding nonce...")
        attempts = 0
        
        while True:
            nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            data = self.transaction_string() + nonce
            h = hashlib.sha256(data.encode()).hexdigest()
            attempts += 1
            
            if h[-1] in "01234":
                print(f"[Block {self.depth}] Found valid nonce after {attempts} attempts!")
                print(f"[Block {self.depth}] Nonce: {nonce}")
                print(f"[Block {self.depth}] Hash: {h}")
                print(f"[Block {self.depth}] Hash Pointer: {self.prev_hash}")
                return nonce
            
            if attempts % 1000 == 0:
                print(f"[Block {self.depth}] Still mining... {attempts} attempts")

    def calculate_hash(self):
        data = self.transaction_string() + self.nonce
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_nonce(self):
        data = self.transaction_string() + self.nonce
        h = hashlib.sha256(data.encode()).hexdigest()
        return h == self.hash and h[-1] in "01234"

    def to_dict(self):
        return {
            'transaction': self.transaction,
            'prev_hash': self.prev_hash,
            'depth': self.depth,
            'nonce': self.nonce,
            'hash': self.hash,
            'is_decided': self.is_decided
        }

    @classmethod
    def from_dict(cls, data):
        block = cls(
            sender=data['transaction']['sender'],
            receiver=data['transaction']['receiver'],
            amount=data['transaction']['amount'],
            prev_hash=data['prev_hash'],
            depth=data['depth'],
            nonce=data['nonce']
        )
        block.is_decided = data.get('is_decided', False)
        return block

    def __str__(self):
        status = "DECIDED" if self.is_decided else "TENTATIVE"
        return (f"Block[{self.depth}] {status}: "
                f"P{self.transaction['sender']} to P{self.transaction['receiver']} "
                f"${self.transaction['amount']}")


class Blockchain:
    def __init__(self, node_id):
        self.node_id = node_id
        self.chain = []
        self.lock = threading.Lock()
        self.create_genesis_block()

    def create_genesis_block(self):
        print(f"[Node P{self.node_id}] Creating genesis block...")
        genesis = Block(
            sender=0,
            receiver=0,
            amount=0,
            prev_hash="0" * 64,
            depth=0,
            nonce="3"  
        )
        genesis.is_decided = True
        self.chain.append(genesis)
        
        print(f"[Node P{self.node_id}] Genesis block created!")
        print(f"[Node P{self.node_id}] Genesis hash: {genesis.hash} (ends in '{genesis.hash[-1]}')")

    def add_block(self, sender, receiver, amount, auto_decide=False):
        with self.lock:
            prev_block = self.chain[-1]
            prev_data = prev_block.transaction_string() + prev_block.nonce + prev_block.hash
            hash_pointer = hashlib.sha256(prev_data.encode()).hexdigest()
            
            new_block = Block(
                sender=sender,
                receiver=receiver,
                amount=amount,
                prev_hash=hash_pointer,
                depth=len(self.chain)
            )
            
            if auto_decide:
                new_block.is_decided = True
            
            self.chain.append(new_block)
            return new_block
    
    def get_depth(self):
        with self.lock:
            return len(self.chain)
    
    def get_last_block(self):
        with self.lock:
            return self.chain[-1] if self.chain else None

    def verify_chain(self):
        with self.lock:
            for i in range(len(self.chain)):
                block = self.chain[i]
                
                if not block.verify_nonce():
                    return False
                
                if i > 0:
                    prev = self.chain[i-1]
                    expected_ptr = hashlib.sha256(
                        (prev.transaction_string() + prev.nonce + prev.hash).encode()
                    ).hexdigest()
                    
                    if block.prev_hash != expected_ptr:
                        return False
            return True

    def print_chain(self):
        with self.lock:
            print(f"\n{'='*80}")
            print(f"BLOCKCHAIN - Node P{self.node_id} (Length: {len(self.chain)} blocks)")
            print(f"{'='*80}")
            for block in self.chain:
                print(block)
            print(f"{'='*80}\n")

    def save_to_disk(self):
        with self.lock:
            filename = f"blockchain_node{self.node_id}.json"
            with open(filename, 'w') as f:
                json.dump([block.to_dict() for block in self.chain], f, indent=2)

    def load_from_disk(self):
        filename = f"blockchain_node{self.node_id}.json"
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                with self.lock:
                    self.chain = [Block.from_dict(b) for b in data]
            return True
        except FileNotFoundError:
            return False


class BankAccounts:
    def __init__(self, num_nodes=5, initial_balance=100):
        self.accounts = {i+1: initial_balance for i in range(num_nodes)}
        self.lock = threading.Lock()
    
    def can_transfer(self, sender, receiver, amount):
        with self.lock:
            if sender not in self.accounts or receiver not in self.accounts:
                return False
            return self.accounts[sender] >= amount
    
    def transfer(self, sender, receiver, amount):
        with self.lock:
            if sender not in self.accounts or receiver not in self.accounts:
                print(f"Invalid account(s): P{sender} or P{receiver}")
                return False

            if amount < 0:
                print(f"Stop stealing money")
                return False
            
            if self.accounts[sender] < amount:
                print(f"Insufficient funds: P{sender} has ${self.accounts[sender]}, needs ${amount}")
                return False
            
            self.accounts[sender] -= amount
            self.accounts[receiver] += amount
            print(f"Transfer: P{sender} (${self.accounts[sender]}) to P{receiver} (${self.accounts[receiver]})")
            return True
    
    def get_balance(self, node_id):
        with self.lock:
            return self.accounts.get(node_id, 0)
    
    def print_balances(self):
        with self.lock:
            print("\n" + "="*40)
            print("ACCOUNT BALANCES")
            print("="*40)
            for node_id in sorted(self.accounts.keys()):
                print(f"  P{node_id}: ${self.accounts[node_id]}")
            print("="*40 + "\n")
    
    def save_to_disk(self, node_id):
        with self.lock:
            filename = f"accounts_node{node_id}.json"
            with open(filename, 'w') as f:
                json.dump(self.accounts, f, indent=2)
    
    def load_from_disk(self, node_id):
        filename = f"accounts_node{node_id}.json"
        try:
            with open(filename, 'r') as f:
                self.accounts = {int(k): v for k, v in json.load(f).items()}
            return True
        except FileNotFoundError:
            return False

class NetworkLayer:
    def __init__(self, node_id, config_file='config.json'):
        self.node_id = node_id
        self.config = self.load_config(config_file)
        self.my_address = self.config['nodes'][str(node_id)]
        self.message_queue = Queue()
        self.server_socket = None
        self.is_running = False
        
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.my_address['ip'], self.my_address['port']))
        self.server_socket.listen(5)
        self.is_running = True
        
        print(f"[Network] Server started on {self.my_address['ip']}:{self.my_address['port']}")
        
        server_thread = threading.Thread(target=self._accept_connections, daemon=True)
        server_thread.start()
    
    def _accept_connections(self):
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                handler_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                handler_thread.start()
            except Exception as e:
                if self.is_running:
                    print(f"[Network] Error accepting connection: {e}")
    
    def _handle_client(self, client_socket):
        try:
            data = client_socket.recv(65536)
            if data:
                message = json.loads(data.decode())
                print(f"[Network] ← Received {message['type']} from P{message['from']}")
                self.message_queue.put(message)
        except Exception as e:
            print(f"[Network] Error handling client: {e}")
        finally:
            client_socket.close()
    
    def send_message(self, to_node_id, message):
        def _send():
            try:
                time.sleep(3) 
                
                target = self.config['nodes'][str(to_node_id)]
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5)
                
                client_socket.connect((target['ip'], target['port']))
                
                message['from'] = self.node_id
                
                client_socket.send(json.dumps(message).encode())
                print(f"[Network] to Sent {message['type']} to P{to_node_id}")
                
                client_socket.close()
                return True
                
            except Exception as e:
                print(f"[Network] Failed to send to P{to_node_id}: {e}")
                return False
        
        threading.Thread(target=_send, daemon=True).start()
    
    def broadcast_message(self, message, exclude_self=True):
        for node_id in self.config['nodes'].keys():
            node_id = int(node_id)
            if exclude_self and node_id == self.node_id:
                continue
            self.send_message(node_id, message.copy())
    
    def get_message(self, timeout=None):
        try:
            return self.message_queue.get(timeout=timeout)
        except:
            return None
    
    def stop(self):
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()

class PaxosState:
    def __init__(self, depth):
        self.depth = depth
        self.promised_ballot = None  
        self.accepted_ballot = None 
        self.accepted_value = None   
        self.promises_received = {}  
        self.accepts_received = set()  

class PaxosNode:
    def __init__(self, node_id, network, blockchain, accounts):
        self.node_id = node_id
        self.network = network
        self.blockchain = blockchain
        self.accounts = accounts
        
        self.paxos_states = {}  
        self.lock = threading.Lock()
        
        self.current_proposal = None  
        self.seq_num = 0
    
    def get_paxos_state(self, depth):
        if depth not in self.paxos_states:
            self.paxos_states[depth] = PaxosState(depth)
        return self.paxos_states[depth]
    
    def make_ballot(self, depth):
        with self.lock:
            self.seq_num += 1
            return (self.seq_num, self.node_id, depth)
    
    def compare_ballots(self, ballot1, ballot2):
        if ballot1 is None:
            return False
        if ballot2 is None:
            return True
        return ballot1 > ballot2
    
    def propose_block(self, sender, receiver, amount):
        
        if not self.accounts.can_transfer(sender, receiver, amount):
            print(f"[Paxos] Cannot propose invalid transaction")
            return
        
        depth = self.blockchain.get_depth()
        
        print(f"\n[Paxos] Starting proposal for depth {depth}")
        print(f"[Paxos] Transaction: P{sender} to P{receiver} ${amount}")
        
        ballot = self.make_ballot(depth)
        print(f"[Paxos] Phase 1: PREPARE with ballot {ballot}")
        
        state = self.get_paxos_state(depth)
        state.promises_received = {}
        
        prepare_msg = {
            'type': 'PREPARE',
            'ballot': ballot,
            'depth': depth
        }
        self.network.broadcast_message(prepare_msg, exclude_self=False)
        
        self.current_proposal = {
            'sender': sender,
            'receiver': receiver,
            'amount': amount,
            'depth': depth,
            'ballot': ballot,
            'block': None  
        }
        
        def check_prepare_timeout():
            time.sleep(10) 
            if self.current_proposal and self.current_proposal['ballot'] == ballot:
                if len(state.promises_received) < 3: 
                    print(f"[Paxos] PREPARE timeout - only got {len(state.promises_received)} promises")
                    self.current_proposal = None
        
        threading.Thread(target=check_prepare_timeout, daemon=True).start()
    
    def handle_prepare(self, msg):
        ballot = tuple(msg['ballot'])
        depth = msg['depth']
        from_node = msg['from']
        
        print(f"[Paxos] Received PREPARE from P{from_node} with ballot {ballot}")
        
        if depth < self.blockchain.get_depth():
            print(f"[Paxos] Rejecting PREPARE - depth {depth} < current depth {self.blockchain.get_depth()}")
            return
        
        state = self.get_paxos_state(depth)

        if self.compare_ballots(ballot, state.promised_ballot):
            state.promised_ballot = ballot
            
            print(f"[Paxos] Promising ballot {ballot}")

            promise_msg = {
                'type': 'PROMISE',
                'ballot': ballot,
                'depth': depth,
                'accepted_ballot': state.accepted_ballot,
                'accepted_value': state.accepted_value.to_dict() if state.accepted_value else None
            }
            self.network.send_message(from_node, promise_msg)
        else:
            print(f"[Paxos] Rejecting PREPARE - already promised {state.promised_ballot}")
    
    def handle_promise(self, msg):
        ballot = tuple(msg['ballot'])
        depth = msg['depth']
        from_node = msg['from']
        accepted_ballot = tuple(msg['accepted_ballot']) if msg['accepted_ballot'] else None
        accepted_value = msg['accepted_value']
        
        print(f"[Paxos] Received PROMISE from P{from_node}")

        if not self.current_proposal or tuple(self.current_proposal['ballot']) != ballot:
            print(f"[Paxos] Ignoring PROMISE - not for current proposal")
            return
        
        state = self.get_paxos_state(depth)
        state.promises_received[from_node] = (ballot, accepted_ballot, accepted_value)
        
        print(f"[Paxos] Have {len(state.promises_received)}/5 promises")

        if len(state.promises_received) >= 3 and self.current_proposal.get('block') is None:
            print(f"[Paxos] Got majority of promises!")
            
            highest_accepted = None
            highest_ballot = None
            
            for node, (prom_ballot, acc_ballot, acc_val) in state.promises_received.items():
                if acc_ballot and self.compare_ballots(acc_ballot, highest_ballot):
                    highest_ballot = acc_ballot
                    highest_accepted = acc_val
            
            if highest_accepted:
                print(f"[Paxos] Must propose previously accepted value")
                block = Block.from_dict(highest_accepted)
            else:
                print(f"[Paxos] Mining new block...")
                prev_block = self.blockchain.get_last_block()
                prev_data = prev_block.transaction_string() + prev_block.nonce + prev_block.hash
                hash_pointer = hashlib.sha256(prev_data.encode()).hexdigest()
                
                block = Block(
                    sender=self.current_proposal['sender'],
                    receiver=self.current_proposal['receiver'],
                    amount=self.current_proposal['amount'],
                    prev_hash=hash_pointer,
                    depth=depth,
                    skip_mining=False
                )
            
            self.current_proposal['block'] = block

            print(f"[Paxos] Phase 2: ACCEPT with ballot {ballot}")
            
            accept_msg = {
                'type': 'ACCEPT',
                'ballot': ballot,
                'depth': depth,
                'block': block.to_dict()
            }
            self.network.broadcast_message(accept_msg, exclude_self=False)
    
    def handle_accept(self, msg):
        ballot = tuple(msg['ballot'])
        depth = msg['depth']
        from_node = msg['from']
        block_data = msg['block']
        
        print(f"[Paxos] Received ACCEPT from P{from_node}")
        
        state = self.get_paxos_state(depth)

        if self.compare_ballots(ballot, state.promised_ballot) or ballot == state.promised_ballot:
            state.accepted_ballot = ballot
            state.accepted_value = Block.from_dict(block_data)
            
            print(f"[Paxos] Accepted ballot {ballot}")

            accepted_msg = {
                'type': 'ACCEPTED',
                'ballot': ballot,
                'depth': depth
            }
            self.network.send_message(from_node, accepted_msg)
        else:
            print(f"[Paxos] Rejecting ACCEPT - promised {state.promised_ballot}")
    
    def handle_accepted(self, msg):
        ballot = tuple(msg['ballot'])
        depth = msg['depth']
        from_node = msg['from']
        
        print(f"[Paxos] Received ACCEPTED from P{from_node}")
        
        if not self.current_proposal or tuple(self.current_proposal['ballot']) != ballot:
            print(f"[Paxos] Ignoring ACCEPTED - not for current proposal")
            return
        
        state = self.get_paxos_state(depth)
        state.accepts_received.add(from_node)
        
        print(f"[Paxos] Have {len(state.accepts_received)}/5 accepts")

        if len(state.accepts_received) >= 3:
            print(f"[Paxos] CONSENSUS REACHED!")
            
            block = self.current_proposal['block']
            block.is_decided = True
            
            with self.blockchain.lock:
                self.blockchain.chain.append(block)
            
            self.accounts.transfer(
                block.transaction['sender'],
                block.transaction['receiver'],
                block.transaction['amount']
            )
            
            self.blockchain.save_to_disk()
            self.accounts.save_to_disk(self.node_id)
            
            decision_msg = {
                'type': 'DECISION',
                'ballot': ballot,
                'depth': depth,
                'block': block.to_dict()
            }
            self.network.broadcast_message(decision_msg, exclude_self=True)
            
            self.current_proposal = None
            print(f"[Paxos] Block committed at depth {depth}\n")
    
    def handle_decision(self, msg):
        depth = msg['depth']
        block_data = msg['block']
        from_node = msg['from']
        
        print(f"[Paxos] Received DECISION from P{from_node} for depth {depth}")
        
        current_depth = self.blockchain.get_depth()
        
        if depth == current_depth:
            block = Block.from_dict(block_data)
            block.is_decided = True
            
            with self.blockchain.lock:
                self.blockchain.chain.append(block)
            
            self.accounts.transfer(
                block.transaction['sender'],
                block.transaction['receiver'],
                block.transaction['amount']
            )
            
            self.blockchain.save_to_disk()
            self.accounts.save_to_disk(self.node_id)
            
            print(f"[Paxos] Committed block at depth {depth}")
        
        elif depth < current_depth:
            with self.blockchain.lock:
                existing_block = self.blockchain.chain[depth]
                
                if not existing_block.is_decided:
                    print(f"[Paxos] Replacing tentative block at depth {depth}")
                    
                    if (existing_block.transaction['sender'] != block_data['transaction']['sender'] or
                        existing_block.transaction['receiver'] != block_data['transaction']['receiver'] or
                        existing_block.transaction['amount'] != block_data['transaction']['amount']):
                        
                        self.accounts.transfer(
                            existing_block.transaction['receiver'],
                            existing_block.transaction['sender'],
                            existing_block.transaction['amount']
                        )
                    
                    decided_block = Block.from_dict(block_data)
                    decided_block.is_decided = True
                    self.blockchain.chain[depth] = decided_block
                    
                    self.accounts.transfer(
                        decided_block.transaction['sender'],
                        decided_block.transaction['receiver'],
                        decided_block.transaction['amount']
                    )
                    
                    if len(self.blockchain.chain) > depth + 1:
                        print(f"[Paxos] Removing {len(self.blockchain.chain) - depth - 1} invalid blocks")
                        self.blockchain.chain = self.blockchain.chain[:depth + 1]
                    
                    self.blockchain.save_to_disk()
                    self.accounts.save_to_disk(self.node_id)
                    
                    print(f"[Paxos] Replaced tentative block with decided block at depth {depth}")
                else:
                    print(f"[Paxos] Block at depth {depth} already decided, ignoring")
        
        else:
            print(f"[Paxos] Warning: Received DECISION for future depth {depth}, current depth is {current_depth}")

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.blockchain = Blockchain(node_id)
        self.accounts = BankAccounts()
        self.network = NetworkLayer(node_id)
        self.paxos = PaxosNode(node_id, self.network, self.blockchain, self.accounts)
        
        self.blockchain.load_from_disk()
        self.accounts.load_from_disk(node_id)
        
        self.network.start_server()
        
        self.running = True
        msg_thread = threading.Thread(target=self._process_messages, daemon=True)
        msg_thread.start()
        
        print(f"\n[Node P{node_id}] Initialized and ready!")
    
    def _process_messages(self):
        while self.running:
            msg = self.network.get_message(timeout=0.1)
            if msg:
                self.handle_network_message(msg)
    
    def handle_network_message(self, msg):
        msg_type = msg['type']
        
        if msg_type == 'PREPARE':
            self.paxos.handle_prepare(msg)
        elif msg_type == 'PROMISE':
            self.paxos.handle_promise(msg)
        elif msg_type == 'ACCEPT':
            self.paxos.handle_accept(msg)
        elif msg_type == 'ACCEPTED':
            self.paxos.handle_accepted(msg)
        elif msg_type == 'DECISION':
            self.paxos.handle_decision(msg)
        elif msg_type == 'SYNC_REQUEST':
            self.handle_sync_request(msg)
        elif msg_type == 'SYNC_RESPONSE':
            self.handle_sync_response(msg)
        else:
            print(f"[Network] Unknown message type: {msg_type}")
    
    def handle_sync_request(self, msg):
        their_depth = msg['my_depth']
        my_depth = self.blockchain.get_depth()
        from_node = msg['from']
        
        print(f"[Sync] P{from_node} requesting sync (their depth: {their_depth}, mine: {my_depth})")
        
        if my_depth > their_depth:
            with self.blockchain.lock:
                missing_blocks = [b.to_dict() for b in self.blockchain.chain[their_depth:]]
            
            sync_response = {
                'type': 'SYNC_RESPONSE',
                'blocks': missing_blocks,
                'accounts': self.accounts.accounts
            }
            self.network.send_message(from_node, sync_response)
            print(f"[Sync] Sent {len(missing_blocks)} blocks to P{from_node}")
    
    def handle_sync_response(self, msg):
        blocks_data = msg['blocks']
        accounts_data = msg['accounts']
        from_node = msg['from']
        
        if not blocks_data:
            print(f"[Sync] Blockchain is up-to-date")
            return
        
        print(f"[Sync] Received {len(blocks_data)} blocks from P{from_node}")
        
        with self.blockchain.lock:
            for block_data in blocks_data:
                block = Block.from_dict(block_data)
                self.blockchain.chain.append(block)
                print(f"[Sync] Added block at depth {block.depth}: {block}")
        
        with self.accounts.lock:
            self.accounts.accounts = {int(k): v for k, v in accounts_data.items()}
        
        self.blockchain.save_to_disk()
        self.accounts.save_to_disk(self.node_id)
        
        print(f"[Sync] Blockchain synchronized! New depth: {self.blockchain.get_depth()}")
        self.accounts.print_balances()
    
    def handle_user_command(self, cmd):
        parts = cmd.strip().split()
        if not parts:
            return True
        
        command = parts[0].lower()
        
        if command == 'moneytransfer':
            if len(parts) != 4:
                print("Usage: moneyTransfer <sender> <receiver> <amount>")
                return True
            
            try:
                sender = int(parts[1])
                receiver = int(parts[2])
                amount = int(parts[3])
                
                if sender != self.node_id:
                    print(f"You can only initiate transfers from P{self.node_id}")
                    return True
                
                self.paxos.propose_block(sender, receiver, amount)
                
            except ValueError:
                print("Invalid arguments")
        
        elif command == 'printblockchain':
            self.blockchain.print_chain()
        
        elif command == 'printbalance':
            self.accounts.print_balances()
        
        elif command == 'save':
            self.blockchain.save_to_disk()
            self.accounts.save_to_disk(self.node_id)
            print("State saved to disk")
        
        elif command == 'verify':
            if self.blockchain.verify_chain():
                print("Blockchain verified!")
            else:
                print("Blockchain verification failed!")
        
        elif command == 'failprocess':
            print("\n FAILING NODE - Simulating crash...")
            self.blockchain.save_to_disk()
            self.accounts.save_to_disk(self.node_id)
            self.running = False
            self.network.stop()
            print("Node P{} has FAILED (network stopped)".format(self.node_id))
            return True
        
        elif command == 'fixprocess':
            if self.running:
                print("Node is already running")
                return True
            
            print("\n RECOVERING NODE...")
            
            self.blockchain.load_from_disk()
            self.accounts.load_from_disk(self.node_id)
            
            self.network = NetworkLayer(self.node_id)
            self.paxos.network = self.network
            self.network.start_server()
            
            self.running = True
            msg_thread = threading.Thread(target=self._process_messages, daemon=True)
            msg_thread.start()
            
            print("Node P{} RECOVERED!".format(self.node_id))
            print("Blockchain depth: {}".format(self.blockchain.get_depth()))
            self.accounts.print_balances()
            
            print("\n[Recovery] Checking if blockchain is up-to-date...")
            sync_msg = {
                'type': 'SYNC_REQUEST',
                'my_depth': self.blockchain.get_depth()
            }
            import random
            other_nodes = [i for i in range(1, 6) if i != self.node_id]
            target = random.choice(other_nodes)
            self.network.send_message(target, sync_msg)
            
            return True
        
        elif command == 'reset':
            print("\n WARNING: This will delete all blockchain and account data!")
            print("Are you sure? Type 'yes' to confirm: ", end='', flush=True)
            confirm = input().strip().lower()
            if confirm == 'yes':
                import glob
                for f in glob.glob("blockchain_node*.json") + glob.glob("accounts_node*.json"):
                    try:
                        os.remove(f)
                        print(f"Deleted {f}")
                    except:
                        pass
                print("\n Reset complete! Restart all nodes to begin fresh.")
            else:
                print("Reset cancelled.")
        
        elif command == 'help':
            print("\nCommands:")
            print("  moneyTransfer <sender> <receiver> <amount>")
            print("  printBlockchain")
            print("  printBalance")
            print("  verify")
            print("  save")
            print("  failProcess   - Simulate node crash")
            print("  fixProcess    - Recover from crash")
            print("  reset         - Delete all saved blockchain/account data")
            print("  help")
            print("  exit")
        
        elif command == 'exit':
            return False
        
        else:
            print(f"Unknown command: {command}. Type 'help' for commands.")
        
        return True
    
    def run(self):
        print("\nType 'help' for available commands")
        self.accounts.print_balances()
        
        while True:
            try:
                print(f"\n[Node P{self.node_id}]> ", end='', flush=True)
                cmd = input()
                if not self.handle_user_command(cmd):
                    break
                    
            except KeyboardInterrupt:
                print("\n\n Shutting down...")
                break
        
        self.running = False
        self.network.stop()

def main():
    if len(sys.argv) < 2:
        print("Usage: python node.py <node_id>")
        print("Example: python node.py 1")
        sys.exit(1)
    
    try:
        node_id = int(sys.argv[1])
        if node_id < 1 or node_id > 5:
            print("Node ID must be between 1 and 5")
            sys.exit(1)
    except ValueError:
        print("Node ID must be a number")
        sys.exit(1)
    
    node = Node(node_id)
    node.run()

if __name__ == "__main__":
    main()