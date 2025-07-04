const warpNode = require('../target/release/warp.node');

module.warp = new warpNode.Warp();
console.log("warp", module.warp);

export function search(query, threshold) {
    return module.warp.search(query, threshold);
}
