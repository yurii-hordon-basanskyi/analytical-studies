# Ви створюєте пригодницьку гру і використовуєте для зберігання предметів гравця словник, у якому
# ключі - це назви предметів, значення - кількість одиниць кожної із речей. Наприклад, словник може
# виглядати так: things = {&#39;key&#39;: 3, &#39;mace&#39;: 1, &#39;gold coin&#39;: 24, &#39;lantern&#39;: 1,
# &#39;stone&#39;: 10}.

inventory = {
    'key': 3,
    'mace': 1,
    'gold coin': 23,
    'lantern': 1,
    'stone': 10,
    'mysterious grimoire': 1
}

print('Equipment:')
total = 0
for item, quantity in inventory.items():
    print(f"{quantity} {item.capitalize()}")
    total += quantity
print(f'Total number of things: {total}')
