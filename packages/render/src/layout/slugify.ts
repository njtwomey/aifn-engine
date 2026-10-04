import { slug } from 'github-slugger'

/** Generate a URL-friendly slug from text. */
export const slugify = (text: string) => slug(text)
