<template>
    <p-interface :menuOptions="menuOptions">
        <div class="template-card-list">
            <component v-for="card in visStatusStore.visibleCards" :key="card.id" :is="components[card.cardType as compOption]"
                :itemProps="card">
            </component>
        </div>
    </p-interface>
</template>
<script setup lang="ts">
import store, { components, compOption} from "@perspecto-cards/components/store"
import { menuOptions } from "@/utils/menu"
const visStatusStore = store.visStatusStore()
</script>

<style scoped></style>
